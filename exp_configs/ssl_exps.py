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
    'data_root':'/mnt/datasets/public/mini-imagenet/'
}

tiered_imagenet = {
    "dataset": "tiered-imagenet",
    "n_classes": 351,
    "dataset_train": "episodic_tiered-imagenet",
    "dataset_val": "episodic_tiered-imagenet",
    "dataset_test": "episodic_tiered-imagenet",
    'data_root':'/mnt/datasets/public/research/tiered-imagenet'
}

cub = {
    "dataset": "cub",
    "n_classes": 100,
    "dataset_train": "episodic_cub",
    "dataset_val": "episodic_cub",
    "dataset_test": "episodic_cub",
    'data_root':'/mnt/datasets/public/research/CUB_200_2011'
}

EXP_GROUPS = {"ssl": []}

pretrained_weights_root = [
                           
                            'csv',
                            #  'hdf5', 
                            # "/mnt/datasets/public/research/adaptron_laplace/logs_borgy_finetune_haven",
                            ]

exp_id_list = ['6bf568c652fcf9f6c85319dc95050577',  
                '48d3878d57365b7cf3327c1b219ae832', 
                'b12196c25ac9b6b007d6235e60e6aa60',
                'c170b92eb5ca0deef1db7262db651ef0',
                 'f0655d6924e734e557f4f070703a96f7', 
                 'e0754c564e95fca0d3f438ed99a0ed7c', 
                 '48dfaaeccce8d91d5a6f51f62d049126', 
                 'f79a0640a58c7771e652c20ca983cf57', 
                 '91bc7a2853e411e5c20585475c742617',
                  '4eb6e371fc5a022299f8e544dafc9188', 
                '961d5fec36ca842271f4b3e7dcacc2d8', 
                '70c0feb6b17a5fdcc1f674e9f97e48bb']

savedir_base = ('/mnt/projects/vision_prototypes/embedding_propagation/david_logs/')
EXP_GROUPS['ssl'] = []
for dataset in [miniimagenet, cub,  tiered_imagenet,  ]:
    # EXP_GROUPS['ssl_%s' % dataset['dataset']] = []
    for backbone in [resnet12, conv4, wrn]:
        for norm_prop in [1]:
                for shot in [1, 5]:
                    for alpha in [0.2]:
                        for w in pretrained_weights_root:
                            for e in exp_id_list:
                                exp_dict = hu.load_json(os.path.join(savedir_base, 
                                            e, 'exp_dict.json'))
                                if (exp_dict['model']['backbone'] != backbone['backbone'] or 
                                    exp_dict['dataset_train'] != dataset['dataset_train'] or
                                    exp_dict['n_classes'] != dataset["n_classes"] or
                                    exp_dict['support_size_train'] != shot
                                    ):
                                    continue
                                # EXP_GROUPS['ssl_%s' % dataset['dataset']] 
                                EXP_GROUPS['ssl'] += [{
                                    'dataset_train_root': dataset["data_root"],
                                    'dataset_val_root': dataset["data_root"],
                                    'dataset_test_root': dataset["data_root"],
                                                        "model": backbone,
                                                        'checkpoint_exp_id':os.path.join(savedir_base, 
                                                                                e),
                                                            
                                                            # Hardware
                                                            "ngpu": 1,
                                                            "random_seed": 42,

                                                            # Optimization
                                                            "batch_size": 1,
                                                            "train_iters": 10,
                                                            "val_iters": 600,
                                                            "test_iters": 600,
                                                            "tasks_per_batch": 1,
                                                            # "pretrained_weights_root": w,

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
                                                            "predict_method":"double_label_prop",

                                                            # Hparams
                                                            "embedding_prop" : False,
                                                            "few_shot_weight": 1,
                                                            "classification_weight": 0.5,
                                                            "rotation_weight": 0,
                                                            "active_size": 0,
                                                            "distance_type": "labelprop",
                                                            "kernel_type": "rbf",
                                                            "kernel_standarization": "all",
                                                            "kernel_bound": "",
                                                            "labelprop_alpha": alpha, 
                                                            "labelprop_scale": 1,
                                                            "norm_prop": norm_prop,
                                                            "rotation_labels": [0],
                                                            }]

# EXP_GROUPS['ssl_inc'] = []
# for dataset in [miniimagenet ]:
#     # EXP_GROUPS['ssl_%s' % dataset['dataset']] = []
#     for backbone in [conv4]:
#         for norm_prop in [1]:
#                 for shot in [1]:
#                     for alpha in [0.2]:
#                         for w in pretrained_weights_root:
#                             for u in [1,2,3,4]:
#                                 for e in exp_id_list:
#                                     exp_dict = hu.load_json(os.path.join(savedir_base, 
#                                                 e, 'exp_dict.json'))
#                                     if (exp_dict['model']['backbone'] != backbone['backbone'] or 
#                                         exp_dict['dataset_train'] != dataset['dataset_train'] or
#                                         exp_dict['n_classes'] != dataset["n_classes"] 
#                                         # exp_dict['support_size_train'] != shot
#                                         ):
#                                         continue
#                                     # EXP_GROUPS['ssl_%s' % dataset['dataset']] 
#                                     EXP_GROUPS['ssl'] += [{
#                                         'dataset_train_root': dataset["data_root"],
#                                         'dataset_val_root': dataset["data_root"],
#                                         'dataset_test_root': dataset["data_root"],
#                                                             "model": backbone,
#                                                             'checkpoint_exp_id':os.path.join(savedir_base, 
#                                                                                     e),
                                                                
#                                                                 # Hardware
#                                                                 "ngpu": 1,
#                                                                 "random_seed": 42,

#                                                                 # Optimization
#                                                                 "batch_size": 1,
#                                                                 "train_iters": 10,
#                                                                 "val_iters": 600,
#                                                                 "test_iters": 600,
#                                                                 "tasks_per_batch": 1,
#                                                                 # "pretrained_weights_root": w,

#                                                                 # Model
#                                                                 "dropout": 0.1,
#                                                                 "avgpool": True,

#                                                                 # Data
#                                                                 'n_classes': dataset["n_classes"],
#                                                                 "collate_fn": "identity",
#                                                                 "transform_train": backbone["transform_train"],
#                                                                 "transform_val": backbone["transform_val"],
#                                                                 "transform_test": backbone["transform_test"],

#                                                                 "dataset_train": dataset["dataset_train"],
#                                                                 "classes_train": 5,
#                                                                 "support_size_train": shot,
#                                                                 "query_size_train": 15,
#                                                                 "unlabeled_size_train": 0,

#                                                                 "dataset_val": dataset["dataset_val"],
#                                                                 "classes_val": 5,
#                                                                 "support_size_val": shot,
#                                                                 "query_size_val": 15,
#                                                                 "unlabeled_size_val": 0,

#                                                                 "dataset_test": dataset["dataset_test"],
#                                                                 "classes_test": 5,
#                                                                 "support_size_test": shot,
#                                                                 "query_size_test": 15,
#                                                                 "unlabeled_size_test": 100,
#                                                                 "predict_method":"double_label_prop",

#                                                                 # Hparams
#                                                                 "embedding_prop" : False,
#                                                                 "few_shot_weight": 1,
#                                                                 "classification_weight": 0.5,
#                                                                 "rotation_weight": 0,
#                                                                 "active_size": 0,
#                                                                 "distance_type": "labelprop",
#                                                                 "kernel_type": "rbf",
#                                                                 "kernel_standarization": "all",
#                                                                 "kernel_bound": "",
#                                                                 "labelprop_alpha": alpha, 
#                                                                 "labelprop_scale": 1,
#                                                                 "norm_prop": norm_prop,
#                                                                 "rotation_labels": [0],
#                                                                 }]

EXP_GROUPS['ssl_tinder'] = []
for dataset in [miniimagenet]:
    for backbone in [conv4, resnet12, ]:
        for norm_prop in [1]:
                for shot, ust in zip([1,2,3,4], [4,3,2,1]):
                    for alpha in [0.2]:
                        for w in ['tinder']:
                            EXP_GROUPS['ssl_tinder'] += [{
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
                                                        "pretrained_weights_root": w,

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
                                                        "predict_method":"double_label_prop",

                                                        # Hparams
                                                        "embedding_prop" : False,
                                                        "few_shot_weight": 1,
                                                        "classification_weight": 0.5,
                                                        "rotation_weight": 0,
                                                        "active_size": 0,
                                                        "distance_type": "labelprop",
                                                        "kernel_type": "rbf",
                                                        "kernel_standarization": "all",
                                                        "kernel_bound": "",
                                                        "labelprop_alpha": alpha, 
                                                        "labelprop_scale": 1,
                                                        "norm_prop": norm_prop,
                                                        "rotation_labels": [0],
                                                        }]