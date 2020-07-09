import os
from haven import haven_utils as hu

import os
from haven import haven_utils as hu

conv4 = {
    "name": "ssl",
    'backbone':'conv4',
    "depth": 4,
    "width": 1,
    "transform_train": "basic",
    "transform_val": "basic",
    "transform_test": "basic"
}

wrn = {
    "name": "ssl",
    "backbone": 'wrn',
    "depth": 28,
    "width": 10,
    "transform_train": "wrn_finetune_train",
    "transform_val": "wrn_val",
    "transform_test": "wrn_val"
}

resnet12 = {
    "name": "ssl",
    "backbone": 'resnet12',
    "depth": 12,
    "width": 1,
    "transform_train": "basic",
    "transform_val": "basic",
    "transform_test": "basic"
}

miniimagenet = {
    "dataset": "miniimagenet",
    "dataset_train": "episodic_miniimagenet",
    "dataset_val": "episodic_miniimagenet",
    "dataset_test": "episodic_miniimagenet",
    "n_classes": 64,
    'data_root':'mini-imagenet/'
}

tiered_imagenet = {
    "dataset": "tiered-imagenet",
    "n_classes": 351,
    "dataset_train": "episodic_tiered-imagenet",
    "dataset_val": "episodic_tiered-imagenet",
    "dataset_test": "episodic_tiered-imagenet",
    'data_root':'tiered-imagenet'
}

cub = {
    "dataset": "cub",
    "n_classes": 100,
    "dataset_train": "episodic_cub",
    "dataset_val": "episodic_cub",
    "dataset_test": "episodic_cub",
    'data_root':'CUB_200_2011'
}

EXP_GROUPS = {}
EXP_GROUPS['ssl_large'] = []
# 12 exps
for dataset in [miniimagenet, tiered_imagenet]:
    for backbone in [resnet12, conv4, wrn]:
        for embedding_prop in [True]:
            for shot in [1, 5]:
                EXP_GROUPS['ssl_large'] += [{
                    'dataset_train_root': dataset["data_root"],
                    'dataset_val_root': dataset["data_root"],
                    'dataset_test_root': dataset["data_root"],
                    "model": backbone,
                        
                    # Hardware
                    "ngpu": 1,
                    "random_seed": 42,

                    # Optimization
                    "batch_size": 1,
                    "train_iters": 10,
                    "test_iters": 600,
                    "tasks_per_batch": 1,

                    # Model
                    "dropout": 0.1,
                    "avgpool": True,

                    # Data
                    'n_classes': dataset["n_classes"],
                    "collate_fn": "identity",
                    "transform_train": backbone["transform_train"],
                    "transform_val": backbone["transform_val"],
                    "transform_test": backbone["transform_test"],

                    "dataset_train": dataset["dataset_train"],
                    "classes_train": 5,
                    "support_size_train": shot,
                    "query_size_train": 15,
                    "unlabeled_size_train": 0,

                    "dataset_val": dataset["dataset_val"],
                    "classes_val": 5,
                    "support_size_val": shot,
                    "query_size_val": 15,
                    "unlabeled_size_val": 0,

                    "dataset_test": dataset["dataset_test"],
                    "classes_test": 5,
                    "support_size_test": shot,
                    "query_size_test": 15,
                    "unlabeled_size_test": 100,
                    "predict_method": "labelprop",
                    "finetuned_weights_root": "./logs/finetuning",

                    # Hparams
                    "embedding_prop" : embedding_prop,
                    }]

# 24 exps
EXP_GROUPS['ssl_small'] = []
for dataset in [tiered_imagenet, miniimagenet]:
    for backbone in [conv4, resnet12, wrn]:
        for embedding_prop in [True]:
                for shot, ust in zip([1,2,3], 
                                     [4,3,2]):
                    EXP_GROUPS['ssl_small'] += [{
                        'dataset_train_root': dataset["data_root"],
                        'dataset_val_root': dataset["data_root"],
                        'dataset_test_root': dataset["data_root"],

                        "model": backbone,
                            
                            # Hardware
                            "ngpu": 1,
                            "random_seed": 42,

                            # Optimization
                            "batch_size": 1,
                            "train_iters": 10,
                            "val_iters": 600,
                            "test_iters": 600,
                            "tasks_per_batch": 1,

                            # Model
                            "dropout": 0.1,
                            "avgpool": True,

                            # Data
                            'n_classes': dataset["n_classes"],
                            "collate_fn": "identity",
                            "transform_train": backbone["transform_train"],
                            "transform_val": backbone["transform_val"],
                            "transform_test": backbone["transform_test"],

                            "dataset_train": dataset["dataset_train"],
                            "classes_train": 5,
                            "support_size_train": shot,
                            "query_size_train": 15,
                            "unlabeled_size_train": 0,

                            "dataset_val": dataset["dataset_val"],
                            "classes_val": 5,
                            "support_size_val": shot,
                            "query_size_val": 15,
                            "unlabeled_size_val": 0,

                            "dataset_test": dataset["dataset_test"],
                            "classes_test": 5,
                            "support_size_test": shot,
                            "query_size_test": 15,
                            "unlabeled_size_test": ust,
                            "predict_method":"labelprop",
                            "finetuned_weights_root": "./logs/finetuning",

                            # Hparams
                            "embedding_prop" : embedding_prop,
                        }]
