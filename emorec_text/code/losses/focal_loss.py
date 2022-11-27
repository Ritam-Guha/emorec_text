import torch
import numpy as np
from emorec_text.config import emotions as emo_orders

def get_weight_per_class() -> torch.tensor:
    emo_cnts={'angry': 2263,
             'disgust': 47,
             'fear': 6576,
             'happy': 3225,
             'sad': 4704,
             'surprise': 566,
             'neutral': 7226,
             'NA': 16203}
    emo_gamma = np.array([0 if k == 'NA' else 1/emo_cnts[k] for k in emo_orders ])
    emo_gamma = torch.from_numpy(emo_gamma)
    return emo_gamma


def focal_loss(ground_truth,
               predictions):
    """
    ref: Focal Loss for Dense Object Detection
         https://arxiv.org/pdf/1708.02002.pdf
         Modified formula (4) for multiclass use case.
         FL(pt) = -(1-pt)^(gamma)*log(pt)
    ref: Implementation of the tensorflow version:
    https://github.com/artemmavrin/focal-loss/blob/master/src/focal_loss/_categorical_focal_loss.py
    """
    # prep work:

    emo_gamma = get_weight_per_class()
    pred_emo_prob = torch.nn.Softmax(dim=-1)(predictions)
    log_pred = torch.log(pred_emo_prob)
    pred_prob_compliment = 1-pred_emo_prob

    fl = -1 * torch.mul(torch.pow(pred_prob_compliment, emo_gamma), log_pred)
    reduced_fl = torch.mean(fl)
    return reduced_fl
