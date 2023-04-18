"""
    Package for Luna-Chess
"""

# Luna classes
from .luna import Luna
from .luna_NN import LunaNN
from .luna_eval import LunaEval
from .luna_dataset import LunaDataset

# Luna constants
LUNA_MAIN_FOLDER = "C:\\Users\\fgoncalves\\dev\\luna"
LUNA_DATA_FOLDER = "data"
CURRENT_MODEL = "main_luna.pth"
NUM_SAMPLES = 2_500_000
from .luna_dataset import DATASET_FOLDER, DATASET_PREFIX 
from .luna_eval import MAXVAL
from .luna_NN import MODEL_FOLDER