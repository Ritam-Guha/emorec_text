from emorec_text.code.training_script.base_training_script import TrainScript
from emorec_text.code.models.rnn import RNNModel
from emorec_text.code.trainers.rnn_trainer import RNNTrainer


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
    train_script = RNNTrainScript(device="cuda")
    train_script.train(n_epochs=300,
                       lr=1e-4)


if __name__ == "__main__":
    main()
