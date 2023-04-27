"""
    file used to generate a Luna-Chess model at will,
    this is supposed to be ran on its own with infinite epochs(ctrl-c to stop training)
"""

import sys
from luna import LunaNN, CURRENT_MODEL

EPOCHS = 5_000_000
VERBOSE = True
CUDA = True
SAVE_AFTER_EACH_EPOCH = True

def infinite_train() -> int:
    """ctrl-c to stop training Luna"""
    LunaNN(model_file=CURRENT_MODEL, verbose=VERBOSE, epochs=EPOCHS, save_after_each_epoch=SAVE_AFTER_EACH_EPOCH)
    
    return 0

if __name__ == "__main__":
    if VERBOSE: print(f"[NEURAL NET] Training Luna Indefinitely... Ctrl-C to stop")
    sys.exit(infinite_train())