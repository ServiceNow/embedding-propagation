from torch.utils.data.dataloader import default_collate

def get_collate(name):
    if name == "identity":
        return lambda x: x
    else:
        return default_collate  