"""
    Luna-Chess Constants
"""

LUNA_MAIN_FOLDER = "D:\luna"
LUNA_DATA_FOLDER = "data"
LUNA_MODEL_FOLDER = "networks"
LUNA_DATASET_FOLDER = "processed" 
LUNA_DATASET_PREFIX = "luna_dataset_" # + number + ".npz"

CURRENT_MODEL = "infinite_luna.pth"
NUM_SAMPLES = 5_000_000

# This just warns the model about if the input will get a pawn struct
INPUT_PAWN_STRUCTURE = False

CUDA = True

SEARCH_DEPTH = 4 # 5 takes too long and 4 is good enough