import torch 
from . import label_prop as lp 
from . import prototypical 
from . import adaptive

def get_predictions(predict_method, episode_dict):
    if predict_method == "labelprop":
        return lp.label_prop_predict(episode_dict)

    elif predict_method == "prototypical":
        return prototypical.prototypical_predict(episode_dict)
    else:
        raise ValueError("Prediction method not found")