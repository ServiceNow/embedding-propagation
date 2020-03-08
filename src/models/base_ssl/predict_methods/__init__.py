import torch 
from . import label_prop as lp 
from . import prototypical 
from . import adaptive

def get_predictions(predict_method, episode_dict):
    if predict_method == "label_prop":
        return lp.label_prop_predict(episode_dict)

    if predict_method == "prototypical":
        return prototypical.prototypical_predict(episode_dict)
    
    if predict_method == "double_label_prop":
        return lp.label_prop_predict(episode_dict, double_flag=True)

    if predict_method == "adaptive":
        return adaptive.adaptive_predict(episode_dict, double_flag=True)