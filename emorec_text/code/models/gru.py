import torch.nn as nn
import torch

import emorec_text.config as config


class GRUModel(torch.nn.Module):
    def __init__(self,
                 device="cpu",
                 input_size=768,
                 hidden_size=256,
                 num_layers=1,
                 num_outputs=len(config.emotions),
                 seed=0,
                 **kwargs):
        super(GRUModel, self).__init__()

        self.device = device
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_outputs = num_outputs
        self.seed = seed

        self.gru = nn.GRU(input_size=self.input_size,
                          hidden_size=self.hidden_size,
                          num_layers=self.num_layers,
                          batch_first=True,
                          bias=True,
                          dropout=0.2,
                          **kwargs)

        self.classifier = nn.Sequential(nn.ReLU(),
                                        nn.Linear(self.hidden_size, self.num_outputs))

    def forward(self, x):
        # initialize the hidden states and cell states
        h_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, dtype=torch.double).to(self.device).to(
            self.device)

        hidden, h_ = self.gru(x, h_0)

        emotions = self.classifier(hidden)

        return emotions

    def load_weights(self,
                     model_path):
        print(f"loading model from epoch: {torch.load(model_path, map_location=self.device)['epoch']}")
        self.load_state_dict(torch.load(model_path, map_location=self.device)["model_state_dict"])
        self.double()


def main():
    model = GRUModel()
    print(model)


if __name__ == "__main__":
    main()
