from emorec_text.code.training_script.base_training_script import TrainScript
from emorec_text.code.models.gru import GRUModel
from emorec_text.code.trainers.gru_trainer import GRUTrainer


class GRUTrainScript(TrainScript):
    def __init__(self,
                 device="cpu"):
        super().__init__(type_model="gru",
                         device=device)
        self.model = None
        self.trainer = None

    def get_model(self):
        self.model = GRUModel(device=self.device).double().to(self.device)
        print(self.model)

    def get_trainer(self):
        self.trainer = GRUTrainer(type_model="gru",
                                  device=self.device)


def main():
    train_script = GRUTrainScript(device="cuda")
    train_script.train()


if __name__ == "__main__":
    main()
