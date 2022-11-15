from emorec_text.code.training_script.base_training_script import TrainScript
from emorec_text.code.models.linear_classifier import LinearClassifierModel
from emorec_text.code.trainers.linear_classifier_trainer import LinearClassifierTrainer


class LinearClassifierTrainScript(TrainScript):
    def __init__(self,
                 device="cpu"):
        super().__init__(type_model="linear_classifier",
                         device=device)
        self.model = None
        self.trainer = None

    def get_model(self):
        self.model = LinearClassifierModel(device=self.device).double().to(self.device)
        print(self.model)

    def get_trainer(self):
        self.trainer = LinearClassifierTrainer(type_model="linear_classifier",
                                               device=self.device)


def main():
    train_script = LinearClassifierTrainScript(device="cuda")
    train_script.train(n_epochs=300,
                       lr=1e-3)


if __name__ == "__main__":
    main()
