import torch


def mse_loss_function(ground_truth,
                      predictions):
    loss = torch.nn.MSELoss(reduction='mean')
    output = loss(predictions, ground_truth)

    return output
