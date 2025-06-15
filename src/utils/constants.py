import numpy as np

EXPECTED_EDGE_TYPES = [
    ('user', 'uses', 'pc'),
    ('user', 'visits', 'url'),
    ('user', 'interacts', 'user')
]

GRAPHSAGE_PARAMS = {
    "hidden_dim": 64,
    "out_dim": 2,
    "num_layers": 1
}

GRAPHSVX_NUM_SAMPLES = 30
DEFAULT_THRESHOLD = 0.15
THRESHOLDS = [round(x, 2) for x in np.arange(0, 1.05, 0.05)]
MAX_BAR_LEN = 20
EPOCHS = 100
LR = 0.01
WEIGHT_DECAY = 5e-4
CLASS_WEIGHT_SCALING = 2
CLASS_NAMES = ['Normal', 'Insider']
CLASS_LABELS = [0, 1]