import torch


def cross_entropy_loss_function(ground_truth,
                                predictions):
    """
    Input:
    predictions: the predictions layers of LSTM model, which is 1xN embedding vector
    ground_truth: the 1xN ground_truch vector correspond to the embeddings.
    Output of this function:
    tensor(5.2485, grad_fn=<NllLossBackward0>)
    """

    # taking log b/c torch.nn.CrossEntropyLoss() assumes
    # logit input
    predictions = torch.log(predictions)

    loss = torch.nn.CrossEntropyLoss(reduction='mean')
    output = loss(predictions, ground_truth)

    return output
