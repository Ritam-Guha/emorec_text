import torch.nn as nn
import torch

import emorec_text.config as config


class LinearClassifierModel(torch.nn.Module):
    def __init__(self,
                 device="cpu",
                 input_size=768,
                 num_outputs=len(config.emotions),
                 seed=0,
                 **kwargs):
        super(LinearClassifierModel, self).__init__()

        # take the necessary user-specified parameters
        self.device = device
        self.input_size = input_size
        self.num_outputs = num_outputs
        self.seed = seed

        # linear classifier
        self.classifier = nn.Sequential(nn.ReLU(),
                                        nn.Linear(self.input_size, self.num_outputs))

    def forward(self, x):
        # use the linear layer to map the embeddings directly to emotions
        emotions = self.classifier(x)

        return emotions

    def load_weights(self,
                     model_path):
        # load the saved weights for the model
        print(f"loading model from epoch: {torch.load(model_path, map_location=self.device)['epoch']}")
        self.load_state_dict(torch.load(model_path, map_location=self.device)["model_state_dict"])
        self.double()


def main():
    model = LinearClassifierModel()
    print(model)


if __name__ == "__main__":
    main()
