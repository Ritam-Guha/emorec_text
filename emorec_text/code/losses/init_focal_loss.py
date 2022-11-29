from emorec_text.config import emotions as emo_orders
import torch
import numpy as np

def get_weight_per_class() -> torch.tensor:
    emo_cnts={'angry': 2263,
             'disgust': 47,
             'fear': 6576,
             'happy': 3225,
             'sad': 4704,
             'surprise': 566,
             'neutral': 7226,
             'NA': 16203}
    emo_gamma = np.array([0 if k == 'NA' else 1/emo_cnts[k] for k in emo_orders ])
    emo_gamma = torch.from_numpy(emo_gamma)
    return emo_gamma

emo_alpha = get_weight_per_class()
focal_loss = torch.hub.load(
	'adeelh/pytorch-multi-class-focal-loss',
	model='FocalLoss',
	alpha=emo_alpha,
	gamma=3,
	reduction='mean',
	force_reload=True
)
