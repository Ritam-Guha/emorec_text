from emorec_text.code.training_script.base_training_script import TrainScript
from emorec_text.code.models.rnn import RNNModel
from emorec_text.code.trainers.rnn_trainer import RNNTrainer

import argparse

parser = argparse.ArgumentParser("rnn_training_script")
parser.add_argument("--device", type=str, default="cpu", help="select the device")
parser.add_argument("--lr", type=float, default=0.0001, help="select the learning rate")
parser.add_argument("--n_epochs", type=int, default=300, help="select the number of epochs")
parser.add_argument("--type_loss", type=str, default="mse", help="select the type of loss")


class RNNTrainScript(TrainScript):
    def __init__(self,
                 device="cpu"):
        super().__init__(type_model="gru",
                         device=device)
        self.model = None
        self.trainer = None

    def get_model(self):
        self.model = RNNModel(device=self.device).double().to(self.device)
        print(self.model)

    def get_trainer(self):
        self.trainer = RNNTrainer(type_model="rnn",
                                  device=self.device)


def main():
    args = parser.parse_args()
    train_script = RNNTrainScript(device=args.device)
    train_script.train(n_epochs=args.n_epochs,
                       lr=args.lr,
                       type_loss=args.type_loss)


if __name__ == "__main__":
    main()
