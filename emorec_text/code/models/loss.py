import torch
import numpy as np


def get_ground_truth_indices(target):
    """
    Return a 1-D tensor object that contains the indices of
    ground truth labels, which is the input format assumed by
    `torch.nn.CrossEntropyLoss(..)`

    Input:
    `target` is the tensor vector that contains ground truth labels.
    Example:
    target = tensor([[1., 0., 0., 0., 0.],
                    [0., 1., 0., 0., 0.],
                    [0., 0., 1., 0., 0.]])

    Output:
    tensor([0, 1, 2])
    """
    np_target = target.numpy(force=True)
    ground_truth_indices = torch.from_numpy(np.where(np_target == 1)[0])
    return ground_truth_indices


def cross_entropy_loss_function(hidden, ground_truth):
    """
    Input:
    hidden: the hidden layers of LSTM model, which is 1xN embedding vector
    ground_truth: the 1xN ground_truch vector correspond to the embeddings.
    Output of this function:
    tensor(5.2485, grad_fn=<NllLossBackward0>)
    """

    ground_truth_indices = get_ground_truth_indices(ground_truth)
    # taking log b/c torch.nn.CrossEntropyLoss() assumes
    # logit input
    hidden = torch.log(hidden)

    loss = torch.nn.CrossEntropyLoss(reduction='sum')
    output = loss(hidden, ground_truth_indices)
    output.backward()

    return output


def mse_loss_function(ground_truth,
                      hidden):
    loss = torch.nn.MSELoss(reduction='sum')
    output = loss(hidden, ground_truth)

    return output


def get_loss(type_loss="mse"):
    dict_loss = {
        "mse": mse_loss_function,
        "cross_entropy": cross_entropy_loss_function
    }
    return dict_loss[type_loss]
