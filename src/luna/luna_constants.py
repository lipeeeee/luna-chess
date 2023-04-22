"""
    Luna-Chess Constants
"""

LUNA_MAIN_FOLDER = "D:\luna"
LUNA_DATA_FOLDER = "data"
LUNA_MODEL_FOLDER = "networks"
LUNA_DATASET_FOLDER = "processed" 
LUNA_DATASET_PREFIX = "luna_dataset_" # + number + ".npz"

CURRENT_MODEL = "serialize_stockfish_refacto.pth"
NUM_SAMPLES = 5_000_000

CUDA = True

SEARCH_DEPTH = 3 # 5 takes too long and 4 is good enough