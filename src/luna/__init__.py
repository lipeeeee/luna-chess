"""
    Package for Luna-Chess
"""

# Luna classes
from .luna import Luna
from .luna_NN import LunaNN
from .luna_eval import LunaEval
from .luna_dataset import LunaDataset

# Luna constants
from .luna_dataset import DATASET_FOLDER, DATASET_PREFIX 
from .luna_eval import MAXVAL
from .luna_NN import MODEL_FOLDER
from .luna_constants import *