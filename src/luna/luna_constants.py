"""
    Luna-Chess Constants
"""

LUNA_MAIN_FOLDER = "D:\luna"
LUNA_DATA_FOLDER = "data"
LUNA_MODEL_FOLDER = "networks"
LUNA_DATASET_FOLDER = "processed" 
LUNA_DATASET_PREFIX = "luna_dataset_" # + number + ".npz"

CURRENT_MODEL = "perfect_luna_maybe_lol_uint8_5m_pool_drouput.pth"
NUM_SAMPLES = 5_000_000

CUDA = True

SEARCH_DEPTH = 2 # 5 takes too long and 4 is good enough