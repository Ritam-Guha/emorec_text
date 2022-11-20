from emorec_text.code.evaluation.rnn_evaluation import RNNEvaluator
from emorec_text.code.evaluation.lstm_evaluation import LSTMEvaluator
from emorec_text.code.evaluation.gru_evaluation import GRUEvaluator
from emorec_text.code.evaluation.linear_classifier_evaluation import LinearClassifierEvaluator


evaluator_dict = {
    "rnn": RNNEvaluator,
    "lstm": LSTMEvaluator,
    "gru": GRUEvaluator,
    "linear_classifier": LinearClassifierEvaluator
}


def get_evaluator(type_model):
    return evaluator_dict[type_model]
