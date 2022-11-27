import torch
import numpy as np

def get_ground_truth_indices(target):
    """
    Return a 1-D tensor object that contains the indices of
    ground truth lables, which is the input format assumed by
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
    ground_truth_indices = torch.from_numpy(np.where(np_target == 1)[1])
    return ground_truth_indices

def cross_entropy_loss_function(ground_truth,
                                predictions):
    """
    Input:
    predictions: the predictions layers of LSTM model, which is 1xN embedding vector
    ground_truth: the 1xN ground_truch vector correspond to the embeddings.
    Output of this function:
    tensor(5.2485, grad_fn=<NllLossBackward0>)
    """
    ground_truth_indices = get_ground_truth_indices(ground_truth)

    predictions = torch.nn.Softmax(dim=-1)(predictions)

    # taking log b/c torch.nn.CrossEntropyLoss() assumes
    # logit input
    predictions = torch.log(predictions)

    loss = torch.nn.CrossEntropyLoss(reduction='mean')
    output = loss(predictions, ground_truth_indices)
    # print(output.requires_grad)

    return output
