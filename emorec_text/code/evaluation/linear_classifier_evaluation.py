from emorec_text.code.evaluation.base_evaluation import Evaluator
from emorec_text.code.models.linear_classifier import LinearClassifierModel


class LinearClassifierEvaluator(Evaluator):
    def __init__(self,
                 lr=0.0001,
                 n_epochs=100):
        super().__init__(type_model="linear_classifier",
                         lr=lr,
                         n_epochs=n_epochs)
        self.model = LinearClassifierModel()


def main():
    evaluator = LinearClassifierEvaluator()
    acc = evaluator.evaluate()


if __name__ == "__main__":
    main()