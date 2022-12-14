from emorec_text.code.evaluation.base_evaluation import Evaluator
from emorec_text.code.models.gru import GRUModel


class GRUEvaluator(Evaluator):
    def __init__(self,
                 lr=0.0001,
                 n_epochs=500):
        super().__init__(type_model="gru",
                         lr=lr,
                         n_epochs=n_epochs)
        self.model = GRUModel().eval()


def main():
    evaluator = GRUEvaluator()
    acc = evaluator.evaluate()


if __name__ == "__main__":
    main()