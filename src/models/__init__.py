import torch
import tqdm
import argparse
import pandas as pd
import pickle, os
import numpy as np
from . import pretraining, finetuning, ssl_wrapper

def get_model(model_name, backbone, n_classes, exp_dict, pretrained_weights_dir=None, savedir_base=None):
    if model_name == "pretraining":
        model = pretraining.PretrainWrapper(backbone, n_classes, exp_dict)

    elif model_name == "finetuning":
        model =  finetuning.FinetuneWrapper(backbone, n_classes, exp_dict)
        
    elif model_name == "ssl":
        model =  ssl_wrapper.SSLWrapper(backbone, n_classes, exp_dict, savedir_base=savedir_base)

    else:
        raise ValueError('model does not exist...')

    # load pretrained model
    if pretrained_weights_dir:
        s_path = os.path.join(os.path.dirname(pretrained_weights_dir), 'score_list_best.pkl')
        if not os.path.exists(s_path):
            s_path = os.path.join(exp_dict['checkpoint_exp_id'], 
                                       'score_list.pkl')
        print('Loaded checkpoint from exp_id: %s' % 
          os.path.split(os.path.dirname(pretrained_weights_dir))[-1]
        )
        print('Fine-tuned accuracy: %.3f' % hu.load_pkl(s_path)[-1]['test_accuracy'])
        model.model.load_state_dict(torch.load(pretrained_weights_dir)['model'])

    return model 

# ===============================================
# Trainers
def train_on_loader(model, train_loader):
    model.train()

    n_batches = len(train_loader)
    train_monitor = TrainMonitor()
    for e in range(1):
        for i, batch in enumerate(train_loader):
            score_dict = model.train_on_batch(batch)
            
            train_monitor.add(score_dict)
            if i % 10 == 0:
                msg = "%d/%d %s" % (i, n_batches, train_monitor.get_avg_score())
                
                print(msg)

    return train_monitor.get_avg_score()

def val_on_loader(model, val_loader, val_monitor):
    model.eval()

    n_batches = len(val_loader)
    
    for i, batch in enumerate(val_loader):
        score = model.val_on_batch(batch)

        val_monitor.add(score)
        if i % 10 == 0:
            msg = "%d/%d %s" % (i, n_batches, val_monitor.get_avg_score())
            
            print(msg)


    return val_monitor.get_avg_score()


@torch.no_grad()
def vis_on_loader(model, vis_loader, savedir):
    model.eval()

    n_batches = len(vis_loader)
    split = vis_loader.dataset.split
    for i, batch in enumerate(vis_loader):
        print("%d - visualizing %s image - savedir:%s" % (i, batch["meta"]["split"][0], savedir.split("/")[-2]))
        model.vis_on_batch(batch, savedir=savedir)
        

def test_on_loader(model, test_loader):
    model.eval()
    ae = 0.
    n_samples = 0.

    n_batches = len(test_loader)
    pbar = tqdm.tqdm(total=n_batches)
    for i, batch in enumerate(test_loader):
        pred_count = model.predict(batch, method="counts")

        ae += abs(batch["counts"].cpu().numpy().ravel() - pred_count.ravel()).sum()
        n_samples += batch["counts"].shape[0]

        pbar.set_description("TEST mae: %.4f" % (ae / n_samples))
        pbar.update(1)

    pbar.close()
    score = ae / n_samples
    print({"test_score": score, "test_mae":score})

    return {"test_score": score, "test_mae":score}


class TrainMonitor:
    def __init__(self):
        self.score_dict_sum = {}
        self.n = 0

    def add(self, score_dict):
        for k,v in score_dict.items():
            if k not in self.score_dict_sum:
                self.score_dict_sum[k] = score_dict[k]
            else:
                self.n += 1
                self.score_dict_sum[k] += score_dict[k]

    def get_avg_score(self):
        return {k:v/(self.n + 1) for k,v in self.score_dict_sum.items()}

    