"""
    file used to generate a Luna-Chess model at will,
    this is supposed to be ran on its own with infinite epochs(ctrl-c to stop training)
"""

from luna import LunaNN

MODEL_NAME = "infinite_luna.pth"
EPOCHS = 20_000_000
VERBOSE = True
CUDA = True
SAVE_AFTER_EACH_EPOCH = True

def infinite_train() -> None:
    """ctrl-c to stop training Luna"""
    LunaNN(MODEL_NAME, cuda=CUDA, verbose=VERBOSE, epochs=EPOCHS, save_after_each_epoch=SAVE_AFTER_EACH_EPOCH)

if __name__ == "__main__":
    if VERBOSE: print(f"[NEURAL NET] Training Luna Indefinitely... Ctrl-C to stop")
    infinite_train()