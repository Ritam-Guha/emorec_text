from emorec_text.code.evaluation.base_evaluation import Evaluator
from emorec_text.code.models.lstm import LSTMModel


class LSTMEvaluator(Evaluator):
    def __init__(self,
                 lr=0.0001,
                 n_epochs=100):
        super().__init__(type_model="lstm",
                         lr=lr,
                         n_epochs=n_epochs)
        self.model = LSTMModel()


def main():
    evaluator = LSTMEvaluator()
    acc = evaluator.evaluate()


if __name__ == "__main__":
    main()