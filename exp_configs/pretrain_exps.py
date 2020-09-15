from haven import haven_utils as hu

conv4 = {
    "name": "pretraining",
    'backbone':'conv4',
    "depth": 4,
    "width": 1,
    "transform_train": "basic",
    "transform_val": "basic",
    "transform_test": "basic"
}

wrn = {
    "name": "pretraining",
    "backbone": 'wrn',
    "depth": 28,
    "width": 10,
    "transform_train": "wrn_pretrain_train",
    "transform_val": "wrn_val",
    "transform_test": "wrn_val"
}

resnet12 = {
    "name": "pretraining",
    "backbone": 'resnet12',
    "depth": 12,
    "width": 1,
    "transform_train": "basic",
    "transform_val": "basic",
    "transform_test": "basic"
}

miniimagenet = {
    "dataset": "miniimagenet",
    "dataset_train": "rotated_episodic_miniimagenet_pkl",
    "dataset_val": "episodic_miniimagenet_pkl",
    "dataset_test": "episodic_miniimagenet_pkl",
    "n_classes": 64,
    "data_root": "mini-imagenet"
}

tiered_imagenet = {
    "dataset": "tiered-imagenet",
    "n_classes": 351,
    "dataset_train": "rotated_tiered-imagenet",
    "dataset_val": "episodic_tiered-imagenet",
    "dataset_test": "episodic_tiered-imagenet",
    "data_root": "tiered-imagenet",
}

cub = {
    "dataset": "cub",
    "n_classes": 100,
    "dataset_train": "rotated_cub",
    "dataset_val": "episodic_cub",
    "dataset_test": "episodic_cub",
    "data_root": "CUB_200_2011"
}

EXP_GROUPS = {"pretrain": []}

for dataset in [miniimagenet, tiered_imagenet, cub]:
    for backbone in [conv4, resnet12, wrn]:
        for lr in [0.2, 0.1]:
            EXP_GROUPS['pretrain'] += [{"model": backbone,

                                        # Hardware
                                        "ngpu": 4,
                                        "random_seed": 42,

                                        # Optimization
                                        "batch_size": 128,
                                        "target_loss": "val_accuracy",
                                        "lr": lr,
                                        "min_lr_decay": 0.0001,
                                        "weight_decay": 0.0005,
                                        "patience": 10,
                                        "max_epoch": 200,
                                        "train_iters": 600,
                                        "val_iters": 600,
                                        "test_iters": 600,
                                        "tasks_per_batch": 1,

                                        # Model
                                        "dropout": 0.1,
                                        "avgpool": True,

                                        # Data
                                        'n_classes': dataset["n_classes"],
                                        "collate_fn": "default",
                                        "transform_train": backbone["transform_train"],
                                        "transform_val": backbone["transform_val"],
                                        "transform_test": backbone["transform_test"],

                                        "dataset_train": dataset["dataset_train"],
                                        "dataset_train_root": dataset["data_root"],
                                        "classes_train": 5,
                                        "support_size_train": 5,
                                        "query_size_train": 15,
                                        "unlabeled_size_train": 0,

                                        "dataset_val": dataset["dataset_val"],
                                        "dataset_val_root": dataset["data_root"],
                                        "classes_val": 5,
                                        "support_size_val": 5,
                                        "query_size_val": 15,
                                        "unlabeled_size_val": 0,

                                        "dataset_test": dataset["dataset_test"],
                                        "dataset_test_root": dataset["data_root"],
                                        "classes_test": 5,
                                        "support_size_test": 5,
                                        "query_size_test": 15,
                                        "unlabeled_size_test": 0,
                                        

                                        # Hparams
                                        "embedding_prop": True,
                                        "cross_entropy_weight": 1,
                                        "few_shot_weight": 0,
                                        "rotation_weight": 1,
                                        "active_size": 0,
                                        "distance_type": "labelprop",
                                        "kernel_bound": "",
                                        "rotation_labels": [0, 1, 2, 3]
                                        }]
