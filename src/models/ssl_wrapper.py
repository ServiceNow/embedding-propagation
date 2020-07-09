"""
Few-Shot Parallel: trains a model as a series of tasks computed in parallel on multiple GPUs

"""
import copy
import numpy as np
import os
from .base_ssl import oracle
from scipy.stats import sem, t
import torch
import pandas as pd
import torch.nn.functional as F
import tqdm
from src.tools.meters import BasicMeter
from src.modules.distances import standarized_label_prop, _propagate, prototype_distance
from .base_wrapper import BaseWrapper
from haven import haven_utils as hu
import glob
from scipy.stats import sem, t
import shutil as sh
from .base_ssl import selection_methods as sm
from .base_ssl import predict_methods as pm
from embedding_propagation import EmbeddingPropagation

class SSLWrapper(BaseWrapper):
    """Trains a model using an episodic scheme on multiple GPUs"""

    def __init__(self, model, n_classes, exp_dict, pretrained_savedir=None, savedir_base=None):
        """ Constructor
        Args:
            model: architecture to train
            exp_dict: reference to dictionary with the global state of the application
        """
        super().__init__()
        self.model = model
        self.exp_dict = exp_dict 
        self.ngpu = self.exp_dict["ngpu"]
        self.predict_method = exp_dict['predict_method']

        self.model.add_classifier(n_classes, modalities=0)
        self.nclasses = n_classes

        best_accuracy = -1 
        self.label = exp_dict['model']['backbone'] + "_" + exp_dict['dataset_test'].split('_')[1].replace('-imagenet','')
        print('=============')
        print('dataset:', exp_dict["dataset_train"].split('_')[-1]) 
        print('backbone:', exp_dict['model']["backbone"])
        print('n_classes:', exp_dict['n_classes'])
        print('support_size_train:', exp_dict['support_size_train'])

        if pretrained_savedir is None:
            # find the best checkpoint
            savedir_base = exp_dict["finetuned_weights_root"]
            if not os.path.exists(savedir_base):
                raise ValueError("Please set the variable named \
                    'finetuned_weights_root' with the path of the folder \
                    with the episodic finetuning experiments")
            for exp_hash in os.listdir(savedir_base):
                base_path = os.path.join(savedir_base, exp_hash)
                exp_dict_path = os.path.join(base_path, 'exp_dict.json')
                if not os.path.exists(exp_dict_path):
                    continue
                loaded_exp_dict = hu.load_json(exp_dict_path)
                pkl_path = os.path.join(base_path, 'score_list_best.pkl')

                if exp_dict['support_size_train'] in [2,3,4]:
                    support_size_needed = 1
                else:
                    support_size_needed = exp_dict['support_size_train']

                if (loaded_exp_dict["model"]["name"] == 'finetuning' and 
                    loaded_exp_dict["dataset_train"].split('_')[-1] == exp_dict["dataset_train"].split('_')[-1] and 
                    loaded_exp_dict["model"]["backbone"] == exp_dict['model']["backbone"] and
                    loaded_exp_dict['n_classes'] == exp_dict["n_classes"] and
                    loaded_exp_dict['support_size_train'] == support_size_needed,
                    loaded_exp_dict["embedding_prop"] == exp_dict["embedding_prop"]):
                    
                    model_path = os.path.join(base_path, 'checkpoint_best.pth')

                    try:
                        print("Attempting to load ", model_path)
                        accuracy = hu.load_pkl(pkl_path)[-1]["val_accuracy"]
                        self.model.load_state_dict(torch.load(model_path)['model'], strict=False)
                        if accuracy > best_accuracy:
                            best_path = os.path.join(base_path, 'checkpoint_best.pth')
                            best_accuracy = accuracy
                    except Exception as e:
                        print(e)
                   
            assert(best_accuracy > 0.1)
            print("Finetuning %s with original accuracy : %f" %(base_path, best_accuracy))
            self.model.load_state_dict(torch.load(best_path)['model'], strict=False)
        self.best_accuracy = best_accuracy
        self.acc_sum = 0.0
        self.n_count = 0
        self.model.cuda()

    def get_embeddings(self, embeddings, support_size, query_size, nclasses):
        b, c = embeddings.size()
        
        if self.exp_dict["embedding_prop"] == True:
            embeddings = EmbeddingPropagation()(embeddings)
        return embeddings.view(b, c)

    def get_episode_dict(self, batch):
        nclasses = batch["nclasses"]
        support_size = batch["support_size"]
        query_size = batch["query_size"]
        k = (support_size + query_size)
        c = batch["channels"]
        h = batch["height"]
        w = batch["width"]

        tx = batch["support_set"].view(support_size, nclasses, c, h, w).cuda(non_blocking=True)
        vx = batch["query_set"].view(query_size, nclasses, c, h, w).cuda(non_blocking=True)
        ux = batch["unlabeled_set"].view(batch["unlabeled_size"], nclasses, c, h, w).cuda(non_blocking=True)
        x = torch.cat([tx, vx, ux], 0)
        x = x.view(-1, c, h, w).cuda(non_blocking=True)

        if self.ngpu > 1:
            features = self.parallel_model(x, is_support=True)
        else:
            features = self.model(x, is_support=True)

        embeddings = self.get_embeddings(features, 
                                    support_size, 
                                    query_size+
                                    batch['unlabeled_size'], 
                                    nclasses) # (b, channels)
   
        uniques = np.unique(batch['targets'])
        labels = torch.zeros(batch['targets'].shape[0])
        for i, u in enumerate(uniques):
            labels[batch['targets']==u] = i

        ## perform ssl
        # 1. indices
        episode_dict = {}
        ns = support_size*nclasses
        nq = query_size*nclasses
        episode_dict["support"] = {'samples':embeddings[:ns], 
                                   'labels':labels[:ns]}
        episode_dict["query"] = {'samples':embeddings[ns:ns+nq], 
                          'labels':labels[ns:ns+nq]}
        episode_dict["unlabeled"] = {'samples':embeddings[ns+nq:]}
        # batch["support_so_far"] = {'samples':embeddings, 
        #                            'labels':labels}

        
        for k, v in episode_dict.items():
            episode_dict[k]['samples'] = episode_dict[k]['samples'].cpu().numpy()
            if 'labels' in episode_dict[k]:
                episode_dict[k]['labels'] = episode_dict[k]['labels'].cpu().numpy().astype(int)
        return episode_dict

    def predict_on_batch(self, episode_dict, support_size_max=None):
        ind_selected = sm.get_indices(selection_method="ssl",
                                    episode_dict=episode_dict,
                                    support_size_max=support_size_max)
        episode_dict = update_episode_dict(ind_selected, episode_dict)
        pred_labels = pm.get_predictions(predict_method=self.predict_method,
                                         episode_dict=episode_dict)
       
        return pred_labels

    def val_on_batch(self, batch):
        # if self.exp_dict['ora']
        if self.exp_dict.get("pretrained_weights_root") == 'hdf5':
            episode_dict = self.sampler.sample_episode(int(self.exp_dict['support_size_test']), 
                                                            self.exp_dict['query_size_test'], 
                                                            self.exp_dict['unlabeled_size_test'], 
                                                            apply_ten_flag=self.exp_dict.get("apply_ten_flag"))
        else:
            episode_dict = self.get_episode_dict(batch)
        episode_dict["support_so_far"] = copy.deepcopy(episode_dict["support"])
        episode_dict["n_classes"] = 5

        pred_labels = self.predict_on_batch(episode_dict, support_size_max=self.exp_dict['unlabeled_size_test']*self.exp_dict['classes_test'])
        accuracy = oracle.compute_acc(pred_labels=pred_labels, 
                true_labels=episode_dict["query"]["labels"])

        # query_labels = episode_dict["query"]["labels"]
        # accuracy = float((pred_labels == query_labels.cuda()).float().mean())
        
        self.acc_sum += accuracy
        self.n_count += 1
        return -1, accuracy

    @torch.no_grad()
    def test_on_loader(self, data_loader, max_iter=None):
        """Iterate over the validation set

        Args:
            data_loader: iterable validation data loader
            max_iter: max number of iterations to perform if the end of the dataset is not reached
        """
        self.model.eval()

        test_accuracy_meter = BasicMeter.get("test_accuracy").reset()
        test_accuracy = []
        # Iterate through tasks, each iteration loads n tasks, with n = number of GPU
        # dirname = os.path.split(self.exp_dict["pretrained_weights_root"])[-1]
        with tqdm.tqdm(total=len(data_loader)) as pbar:
            for batch_all in data_loader:
                batch = batch_all[0]
                loss, accuracy = self.val_on_batch(batch)

                test_accuracy_meter.update(float(accuracy), 1)
                test_accuracy.append(float(accuracy))

                string = ("'%s'  -  ssl: %.3f" % 
                                (self.label, 
                                # dirname, 
                                test_accuracy_meter.mean()))
                # print(string)
                pbar.update(1)
                pbar.set_description(string)
        
        confidence = 0.95
        n = len(test_accuracy)
        std_err = sem(np.array(test_accuracy))
        h = std_err * t.ppf((1 + confidence) / 2, n - 1)
        return {"test_loss": -1, 
                "ssl_accuracy": test_accuracy_meter.mean(), 
                "ssl_confidence": h,
                'finetuned_accuracy': self.best_accuracy}

def update_episode_dict(ind, episode_dict):
    # 1. update supports so far
    selected_samples = episode_dict["unlabeled"]["samples"][ind]
    selected_labels = episode_dict["unlabeled"]["labels"][ind]
    
    selected_support_dict = {"samples": selected_samples, "labels": selected_labels}

    for k, v in episode_dict["support_so_far"].items():
        episode_dict["support_so_far"][k] = np.concatenate([v, selected_support_dict[k]], axis=0)

    # 2. update unlabeled samples
    n_unlabeled = episode_dict["unlabeled"]["samples"].shape[0]
    ind_rest = np.setdiff1d(np.arange(n_unlabeled), ind)

    new_unlabeled_dict = {}
    for k, v in episode_dict["unlabeled"].items():
        new_unlabeled_dict[k] = v[ind_rest]
    
    episode_dict["unlabeled"] = new_unlabeled_dict

    return episode_dict