from emorec_text.code.losses.mse import mse_loss
from emorec_text.code.losses.cross_entropy import cross_entropy_loss
from emorec_text.code.losses.focal_loss import focal_loss


dict_loss = {
        "mse": mse_loss,
        "cross_entropy": cross_entropy_loss,
        "focal_loss": focal_loss
}


def get_loss(type_loss="mse"):
    return dict_loss[type_loss]
