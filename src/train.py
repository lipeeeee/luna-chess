"""
    file used to generate a Luna-Chess model at will,
    this is supposed to be ran on its own with infinite epochs(ctrl-c to stop training)

    make 
"""

from luna import LunaNN

MODEL_NAME = "infinite_luna.pth"
EPOCHS = 10_000_000
VERBOSE = True
CUDA = True

def infinite_train() -> None:
    """ctrl-c to stop training Luna"""
    LunaNN(MODEL_NAME, cuda=CUDA, verbose=VERBOSE, epochs=EPOCHS)

if __name__ == "__main__":
    if VERBOSE: print(f"[NEURAL NET] Training Luna Indefinitely... Ctrl-C to stop")
    infinite_train()