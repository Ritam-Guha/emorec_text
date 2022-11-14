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

        self.device = device
        self.input_size = input_size
        self.num_outputs = num_outputs
        self.seed = seed

        self.classifier = nn.Sequential(nn.ReLU(),
                                        nn.Linear(self.input_size, self.num_outputs))

    def forward(self, x):
        # initialize the hidden states and cell states
        emotions = self.classifier(x)

        return emotions

    def load_weights(self,
                     model_path):
        print(f"loading model from epoch: {torch.load(model_path, map_location=self.device)['epoch']}")
        self.load_state_dict(torch.load(model_path, map_location=self.device)["model_state_dict"])
        self.double()


def main():
    model = LinearClassifierModel()
    print(model)


if __name__ == "__main__":
    main()
