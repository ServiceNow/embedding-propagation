import torch
import argparse
import pandas as pd
import sys
import os
from torch import nn
from torch.nn import functional as F
import tqdm
import pprint
from src import utils as ut
import torchvision
import numpy as np

from src import datasets, models
from src.models import backbones
from torch.utils.data import DataLoader
import exp_configs
from torch.utils.data.sampler import RandomSampler

from haven import haven_utils as hu
from haven import haven_results as hr
from haven import haven_chk as hc
from haven import haven_jupyter as hj


def trainval(exp_dict, savedir_base, datadir, reset=False, 
            num_workers=0, pretrained_weights_dir=None):
    # bookkeeping
    # ---------------

    # get experiment directory
    exp_id = hu.hash_dict(exp_dict)
    savedir = os.path.join(savedir_base, exp_id)

    if reset:
        # delete and backup experiment
        hc.delete_experiment(savedir, backup_flag=True)
    
    # create folder and save the experiment dictionary
    os.makedirs(savedir, exist_ok=True)
    hu.save_json(os.path.join(savedir, 'exp_dict.json'), exp_dict)
    pprint.pprint(exp_dict)
    print('Experiment saved in %s' % savedir)

    # load datasets
    # ==========================
    train_set = datasets.get_dataset(dataset_name=exp_dict["dataset_train"],
                data_root=os.path.join(datadir, exp_dict["dataset_train_root"]),
                split="train", 
                transform=exp_dict["transform_train"], 
                classes=exp_dict["classes_train"],
                support_size=exp_dict["support_size_train"],
                query_size=exp_dict["query_size_train"], 
                n_iters=exp_dict["train_iters"],
                unlabeled_size=exp_dict["unlabeled_size_train"])

    val_set = datasets.get_dataset(dataset_name=exp_dict["dataset_val"],
                data_root=os.path.join(datadir, exp_dict["dataset_val_root"]),
                split="val", 
                transform=exp_dict["transform_val"], 
                classes=exp_dict["classes_val"],
                support_size=exp_dict["support_size_val"],
                query_size=exp_dict["query_size_val"], 
                n_iters=exp_dict.get("val_iters", None),
                unlabeled_size=exp_dict["unlabeled_size_val"])

    test_set = datasets.get_dataset(dataset_name=exp_dict["dataset_test"],
                data_root=os.path.join(datadir, exp_dict["dataset_test_root"]),
                split="test", 
                transform=exp_dict["transform_val"], 
                classes=exp_dict["classes_test"],
                support_size=exp_dict["support_size_test"],
                query_size=exp_dict["query_size_test"], 
                n_iters=exp_dict["test_iters"],
                unlabeled_size=exp_dict["unlabeled_size_test"])

    # get dataloaders
    # ==========================
    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=exp_dict["batch_size"],
        shuffle=True,
        num_workers=num_workers,
        collate_fn=ut.get_collate(exp_dict["collate_fn"]),
        drop_last=True)
    val_loader = torch.utils.data.DataLoader(
        val_set,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=lambda x: x,
        drop_last=True)
    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=lambda x: x,
        drop_last=True)

    
    # create model and trainer
    # ==========================

    # Create model, opt, wrapper
    backbone = backbones.get_backbone(backbone_name=exp_dict['model']["backbone"], exp_dict=exp_dict)
    model = models.get_model(model_name=exp_dict["model"]['name'], backbone=backbone, 
                                 n_classes=exp_dict["n_classes"],
                                 exp_dict=exp_dict,
                                 pretrained_weights_dir=pretrained_weights_dir,
                                 savedir_base=savedir_base)
    
    # Pretrain or Fine-tune or run SSL
    if exp_dict["model"]['name'] == 'ssl':
        # runs the SSL experiments
        score_list_path = os.path.join(savedir, 'score_list.pkl')
        if not os.path.exists(score_list_path):
            test_dict = model.test_on_loader(test_loader, max_iter=None)
            hu.save_pkl(score_list_path, [test_dict])
        return 
        
    # Checkpoint
    # -----------
    checkpoint_path = os.path.join(savedir, 'checkpoint.pth')
    score_list_path = os.path.join(savedir, 'score_list.pkl')

    if os.path.exists(score_list_path):
        # resume experiment
        model.load_state_dict(hu.torch_load(checkpoint_path))
        score_list = hu.load_pkl(score_list_path)
        s_epoch = score_list[-1]['epoch'] + 1
    else:
        # restart experiment
        score_list = []
        s_epoch = 0

    # Run training and validation
    for epoch in range(s_epoch, exp_dict["max_epoch"]):
        score_dict = {"epoch": epoch}
        score_dict.update(model.get_lr())
        
        # train
        score_dict.update(model.train_on_loader(train_loader))

        # validate
        score_dict.update(model.val_on_loader(val_loader))
        score_dict.update(model.test_on_loader(test_loader))

        # Add score_dict to score_list
        score_list += [score_dict]

        # Report
        score_df = pd.DataFrame(score_list)
        print(score_df.tail())

        # Save checkpoint
        hu.save_pkl(score_list_path, score_list)
        hu.torch_save(checkpoint_path, model.get_state_dict())
        print("Saved: %s" % savedir)

        if "accuracy" in exp_dict["target_loss"]:
            is_best = score_dict[exp_dict["target_loss"]] >= score_df[exp_dict["target_loss"]][:-1].max() 
        else:
            is_best = score_dict[exp_dict["target_loss"]] <= score_df[exp_dict["target_loss"]][:-1].min() 

        # Save best checkpoint
        if is_best:
            hu.save_pkl(os.path.join(savedir, "score_list_best.pkl"), score_list)
            hu.torch_save(os.path.join(savedir, "checkpoint_best.pth"), model.get_state_dict())
            print("Saved Best: %s" % savedir)  
        
        # Check for end of training conditions
        if model.is_end_of_training():
            break


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-e', '--exp_group_list', nargs='+')
    parser.add_argument('-sb', '--savedir_base', required=True)
    parser.add_argument('-d', '--datadir', default='')
    parser.add_argument('-r', '--reset',  default=0, type=int)
    parser.add_argument('-ei', '--exp_id', type=str, default=None)
    parser.add_argument('-j', '--run_jobs', type=int, default=0)
    parser.add_argument('-nw', '--num_workers', default=0, type=int)
    parser.add_argument('-p', '--pretrained_weights_dir', type=str, default=None)

    args = parser.parse_args()

    # Collect experiments
    # -------------------
    if args.exp_id is not None:
        # select one experiment
        savedir = os.path.join(args.savedir_base, args.exp_id)
        exp_dict = hu.load_json(os.path.join(savedir, 'exp_dict.json'))        
        
        exp_list = [exp_dict]
        
    else:
        # select exp group
        exp_list = []
        for exp_group_name in args.exp_group_list:
            exp_list += exp_configs.EXP_GROUPS[exp_group_name]


    # Run experiments or View them
    # ----------------------------
    if args.run_jobs:
        pass
    else:
        # run experiments
        for exp_dict in exp_list:
            # do trainval
            trainval(exp_dict=exp_dict,
                    savedir_base=args.savedir_base,
                    datadir=args.datadir,
                    reset=args.reset,
                    num_workers=args.num_workers,
                    pretrained_weights_dir=args.pretrained_weights_dir)