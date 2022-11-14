from emorec_text.code.evaluation.base_evaluation import Evaluator
from emorec_text.code.models.linear_classifier import LinearClassifierModel
import emorec_text.config as config

import os


class LinearClassifierEvaluator(Evaluator):
    def __init__(self):
        super().__init__(type_model="linear_classifier")
        self.model = LinearClassifierModel()

    def load_model(self):
        path = f"{config.BASE_PATH}/code/model_storage/{self.type_model}/training_best.pt"
        if os.path.exists(path):
            self.model.load_weights(path)


def main():
    evaluator = LinearClassifierEvaluator()
    acc = evaluator.evaluate()


if __name__ == "__main__":
    main()