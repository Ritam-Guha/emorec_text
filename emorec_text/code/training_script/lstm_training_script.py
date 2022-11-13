from emorec_text.code.training_script.base_training_script import TrainScript
from emorec_text.code.models.lstm import LSTMModel
from emorec_text.code.trainers.lstm_trainer import LSTMTrainer


class LSTMTrainScript(TrainScript):
    def __init__(self,
                 device="cpu"):
        super().__init__(type_model="lstm",
                         device=device)
        self.model = None
        self.trainer = None

    def get_model(self):
        self.model = LSTMModel(device=self.device).double().to(self.device)
        print(self.model)

    def get_trainer(self):
        self.trainer = LSTMTrainer(type_model="lstm",
                                   device=self.device)


def main():
    train_script = LSTMTrainScript(device="cuda")
    train_script.train()


if __name__ == "__main__":
    main()


