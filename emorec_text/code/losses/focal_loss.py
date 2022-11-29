import torch
import numpy as np
from emorec_text.code.losses.cross_entropy import get_ground_truth_indices
from emorec_text.code.losses.init_focal_loss import focal_loss as focal_loss_ms


def focal_loss(ground_truth, predictions):
    """
    Focal-loss repo:
    https://github.com/AdeelH/pytorch-multi-class-focal-loss
    """
    ls_losses = [0]*len(ground_truth)
    ls_ground_truths = get_ground_truth_indices(ground_truth)

    for idx in range(len(ground_truth)):
        ground_truth = ls_ground_truths[idx]
        prediction = predictions[idx]
        loss = focal_loss_ms(prediction, ground_truth)
        ls_losses.append(loss)

    ls_losses = torch.tensor(ls_losses, requires_grad=True)
    loss = torch.mean(ls_losses)

    return loss

def focal_loss_og(ground_truth,
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
