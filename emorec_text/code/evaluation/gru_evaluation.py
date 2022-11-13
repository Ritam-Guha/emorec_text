from emorec_text.code.evaluation.base_evaluation import Evaluator
from emorec_text.code.models.gru import GRUModel
import emorec_text.config as config

import os


class GRUEvaluator(Evaluator):
    def __init__(self):
        super().__init__(type_model="gru")
        self.model = GRUModel().eval()

    def load_model(self):
        path = f"{config.BASE_PATH}/code/model_storage/{self.type_model}/training_best.pt"
        if os.path.exists(path):
            self.model.load_weights(path)


def main():
    evaluator = GRUEvaluator()
    acc = evaluator.evaluate()


if __name__ == "__main__":
    main()