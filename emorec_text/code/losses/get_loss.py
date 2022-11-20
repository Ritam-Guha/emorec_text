from emorec_text.code.losses.mse import mse_loss_function
from emorec_text.code.losses.cross_entropy import cross_entropy_loss_function


dict_loss = {
        "mse": mse_loss_function,
        "cross_entropy": cross_entropy_loss_function
}


def get_loss(type_loss="mse"):
    return dict_loss[type_loss]
