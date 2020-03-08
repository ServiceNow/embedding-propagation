
from . import resnet12, conv4, wrn

def get_backbone(backbone_name, exp_dict):
    if backbone_name == "resnet12":
        backbone = resnet12.Resnet12(width=1, dropout=exp_dict["dropout"])
    elif backbone_name == "conv4":
        backbone = conv4.Conv4(exp_dict)
    elif backbone_name == "wrn":
        backbone = wrn.WideResNet(depth=exp_dict["model"]["depth"], width=exp_dict["model"]["width"], exp_dict=exp_dict)

    return backbone
