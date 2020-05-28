"""
Few-Shot Parallel: trains a model as a series of tasks computed in parallel on multiple GPUs

"""
import numpy as np
import os
import torch
import torch.nn.functional as F

from src.tools.meters import BasicMeter
from src.modules.distances import prototype_distance
from embedding_propagation import EmbeddingPropagation, LabelPropagation
from .base_wrapper import BaseWrapper
from haven import haven_utils as haven
from scipy.stats import sem, t
import shutil as sh
import sys


class FinetuneWrapper(BaseWrapper):
    """Finetunes a model using an episodic scheme on multiple GPUs"""

    def __init__(self, model, nclasses, exp_dict):
        """ Constructor
        Args:
            model: architecture to train
            nclasses: number of output classes
            exp_dict: reference to dictionary with the hyperparameters
        """
        super().__init__()
        self.model = model
        self.exp_dict = exp_dict 
        self.ngpu = self.exp_dict["ngpu"]

        self.embedding_propagation = EmbeddingPropagation()
        self.label_propagation = LabelPropagation()
        self.model.add_classifier(nclasses, modalities=0)
        self.nclasses = nclasses

        if self.exp_dict["rotation_weight"] > 0:
            self.model.add_classifier(4, "classifier_rot")

        best_accuracy = -1 
        if self.exp_dict["pretrained_weights_root"] is not None:
            for exp_hash in os.listdir(self.exp_dict['pretrained_weights_root']):
                base_path = os.path.join(self.exp_dict['pretrained_weights_root'], exp_hash)
                exp_dict_path = os.path.join(base_path, 'exp_dict.json')
                if not os.path.exists(exp_dict_path):
                    continue
                loaded_exp_dict = haven.load_json(exp_dict_path)
                pkl_path = os.path.join(base_path, 'score_list_best.pkl')
                if (loaded_exp_dict["model"]["name"] == 'pretraining' and 
                        loaded_exp_dict["dataset_train"].split('_')[-1] == exp_dict["dataset_train"].split('_')[-1] and 
                        loaded_exp_dict["model"]["backbone"] == exp_dict['model']["backbone"] and
                        # loaded_exp_dict["labelprop_alpha"] == exp_dict["labelprop_alpha"] and
                        # loaded_exp_dict["labelprop_scale"] == exp_dict["labelprop_scale"] and
                        os.path.exists(pkl_path)):
                    accuracy = haven.load_pkl(pkl_path)[-1]["val_accuracy"]
                    try:
                        self.model.load_state_dict(torch.load(os.path.join(base_path, 'checkpoint_best.pth'))['model'], strict=False)
                        if accuracy > best_accuracy:
                            best_path = os.path.join(base_path, 'checkpoint_best.pth')
                            best_accuracy = accuracy
                    except:
                        continue
            assert(best_accuracy > 0.1)
            print("Finetuning %s with original accuracy : %f" %(base_path, best_accuracy))
            self.model.load_state_dict(torch.load(best_path)['model'], strict=False)

        # Add optimizers here
        self.optimizer = torch.optim.SGD(self.model.parameters(), 
                                            lr=self.exp_dict["lr"],
                                            momentum=0.9,
                                            weight_decay=self.exp_dict["weight_decay"], 
                                            nesterov=True)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer,
                                                                    mode="min" if "loss" in self.exp_dict["target_loss"] else "max",
                                                                    patience=self.exp_dict["patience"])
        self.model.cuda()
        if self.ngpu > 1:
            self.parallel_model = torch.nn.DataParallel(self.model, device_ids=list(range(self.ngpu)))

    def get_logits(self, embeddings, support_size, query_size, nclasses):
        """Computes the logits from the queries of an episode
        
        Args:
            embeddings (torch.Tensor): episode embeddings
            support_size (int): size of the support set
            query_size (int): size of the query set
            nclasses (int): number of classes
        
        Returns:
            torch.Tensor: logits
        """
        b, c = embeddings.size()

        propagator = None
        if self.exp_dict["embedding_prop"] == True:
            embeddings = self.embedding_propagation(embeddings)

        if self.exp_dict["distance_type"] == "labelprop":
            support_labels = torch.arange(nclasses, device=embeddings.device).view(1, nclasses).repeat(support_size, 1).view(support_size, nclasses)
            unlabeled_labels = nclasses * torch.ones(query_size * nclasses, dtype=support_labels.dtype, device=support_labels.device).view(query_size, nclasses)
            labels = torch.cat([support_labels, unlabeled_labels], 0).view(-1)
            logits = self.label_propagation(embeddings, labels, nclasses)
            logits = logits.view(-1, nclasses, nclasses)[support_size:(support_size + query_size), ...].view(-1, nclasses)

        elif self.exp_dict["distance_tpe"] == "prototypical":
            embeddings = embeddings.view(-1, nclasses, c)
            support_embeddings = embeddings[:support_size]
            query_embeddings = embeddings[support_size:]
            logits = prototype_distance((support_embeddings.view(1, support_size, nclasses, c), False), 
                                        (query_embeddings.view(1, query_size, nclasses, c), False)).view(query_size * nclasses, nclasses)
        return logits

    def train_on_batch(self, batch):
        """Computes the loss on an episode
        
        Args:
            batch (dict): Episode dict
        
        Returns:
            tuple: loss and accuracy of the episode 
        """
        episode = batch[0]
        nclasses = episode["nclasses"]
        support_size = episode["support_size"]
        query_size = episode["query_size"]
        labels = episode["targets"].view(support_size + query_size, nclasses, -1).cuda(non_blocking=True).long()
        k = (support_size + query_size)
        c = episode["channels"]
        h = episode["height"]
        w = episode["width"]

        tx = episode["support_set"].view(support_size, nclasses, c, h, w).cuda(non_blocking=True)
        vx = episode["query_set"].view(query_size, nclasses, c, h, w).cuda(non_blocking=True)
        x = torch.cat([tx, vx], 0)
        x = x.view(-1, c, h, w).cuda(non_blocking=True)
        if self.ngpu > 1:
            embeddings = self.parallel_model(x, is_support=True)
        else:
            embeddings = self.model(x, is_support=True)
        b, c = embeddings.size()

        logits = self.get_logits(embeddings, support_size, query_size, nclasses)
        
        loss = 0
        if self.exp_dict["classification_weight"] > 0:
            loss += F.cross_entropy(self.model.classifier(embeddings.view(b, c)), labels.view(-1)) * self.exp_dict["classification_weight"]

        query_labels = torch.arange(nclasses, device=logits.device).view(1, nclasses).repeat(query_size, 1).view(-1)
        loss += F.cross_entropy(logits, query_labels) * self.exp_dict["few_shot_weight"]
        return loss

    def predict_on_batch(self, batch):
        """Computes the logits of an episode
        
        Args:
            batch (dict): episode dict
        
        Returns:
            tensor: logits for the queries of the current episode
        """
        nclasses = batch["nclasses"]
        support_size = batch["support_size"]
        query_size = batch["query_size"]
        k = (support_size + query_size)
        c = batch["channels"]
        h = batch["height"]
        w = batch["width"]

        tx = batch["support_set"].view(support_size, nclasses, c, h, w).cuda(non_blocking=True)
        vx = batch["query_set"].view(query_size, nclasses, c, h, w).cuda(non_blocking=True)
        x = torch.cat([tx, vx], 0)
        x = x.view(-1, c, h, w).cuda(non_blocking=True)

        if self.ngpu > 1:
            embeddings = self.parallel_model(x, is_support=True)
        else:
            embeddings = self.model(x, is_support=True)

        return self.get_logits(embeddings, support_size, query_size, nclasses) 

    def val_on_batch(self, batch):
        """Computes the loss and accuracy on a validation batch
        
        Args:
            batch (dict): Episode dict
        
        Returns:
            tuple: loss and accuracy of the episode 
        """
        nclasses = batch["nclasses"]
        query_size = batch["query_size"]

        logits = self.predict_on_batch(batch)

        query_labels = torch.arange(nclasses, device=logits.device).view(1, nclasses).repeat(query_size, 1).view(-1)
        loss = F.cross_entropy(logits, query_labels)
        accuracy = float(logits.max(-1)[1].eq(query_labels).float().mean())
        
        return loss, accuracy

    def train_on_loader(self, data_loader, max_iter=None, debug_plot_path=None):
        """Iterate over the training set

        Args:
            data_loader: iterable training data loader
            max_iter: max number of iterations to perform if the end of the dataset is not reached
        """
        self.model.train()
        train_loss_meter = BasicMeter.get("train_loss").reset()
        # Iterate through tasks, each iteration loads n tasks, with n = number of GPU
        self.optimizer.zero_grad()
        for batch_idx, batch in enumerate(data_loader):
            loss = self.train_on_batch(batch) / self.exp_dict["tasks_per_batch"]
            train_loss_meter.update(float(loss), 1)
            loss.backward()
            if ((batch_idx + 1) % self.exp_dict["tasks_per_batch"]) == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()
            if batch_idx + 1 == max_iter:
                break
        return {"train_loss": train_loss_meter.mean()}
        

    @torch.no_grad()
    def val_on_loader(self, data_loader, max_iter=None):
        """Iterate over the validation set

        Args:
            data_loader: iterable validation data loader
            max_iter: max number of iterations to perform if the end of the dataset is not reached
        """
        self.model.eval()
        val_loss_meter = BasicMeter.get("val_loss").reset()
        val_accuracy_meter = BasicMeter.get("val_accuracy").reset()
        # Iterate through tasks, each iteration loads n tasks, with n = number of GPU
        for batch_idx, _data in enumerate(data_loader):
            batch = _data[0]
            loss, accuracy = self.val_on_batch(batch)
            val_loss_meter.update(float(loss), 1)
            val_accuracy_meter.update(float(accuracy), 1)
        loss = BasicMeter.get(self.exp_dict["target_loss"], recursive=True, force=False).mean()
        self.scheduler.step(loss)  # update the learning rate monitor
        return {"val_loss": val_loss_meter.mean(), "val_accuracy": val_accuracy_meter.mean()}

    @torch.no_grad()
    def test_on_loader(self, data_loader, max_iter=None):
        """Iterate over the validation set

        Args:
            data_loader: iterable validation data loader
            max_iter: max number of iterations to perform if the end of the dataset is not reached
        """
        self.model.eval()
        test_loss_meter = BasicMeter.get("test_loss").reset()
        test_accuracy_meter = BasicMeter.get("test_accuracy").reset()
        test_accuracy = []
        # Iterate through tasks, each iteration loads n tasks, with n = number of GPU
        for batch_idx, _data in enumerate(data_loader):
            batch = _data[0]
            loss, accuracy = self.val_on_batch(batch)
            test_loss_meter.update(float(loss), 1)
            test_accuracy_meter.update(float(accuracy), 1)
            test_accuracy.append(float(accuracy))
        from scipy.stats import sem, t
        confidence = 0.95
        n = len(test_accuracy)
        std_err = sem(np.array(test_accuracy))
        h = std_err * t.ppf((1 + confidence) / 2, n - 1)
        return {"test_loss": test_loss_meter.mean(), "test_accuracy": test_accuracy_meter.mean(), "test_confidence": h}

    def get_state_dict(self):
        """Obtains the state dict of this model including optimizer, scheduler, etc
        
        Returns:
            dict: state dict
        """
        ret = {}
        ret["optimizer"] = self.optimizer.state_dict()
        ret["model"] = self.model.state_dict()
        ret["scheduler"] = self.scheduler.state_dict()
        return ret

    def load_state_dict(self, state_dict):
        """Loads the state of the model
        
        Args:
            state_dict (dict): The state to load
        """
        self.optimizer.load_state_dict(state_dict["optimizer"])
        self.model.load_state_dict(state_dict["model"])
        self.scheduler.load_state_dict(state_dict["scheduler"])

    def get_lr(self):
        ret = {}
        for i, param_group in enumerate(self.optimizer.param_groups):
            ret["current_lr_%d" % i] = float(param_group["lr"])
        return ret

    def is_end_of_training(self):
        lr = self.get_lr()["current_lr_0"]
        return lr <= (self.exp_dict["lr"] * self.exp_dict["min_lr_decay"])