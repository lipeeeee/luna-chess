"""
    file used to generate a Luna-Chess model at will,
    this is supposed to be ran on its own with infinite epochs(ctrl-c to stop training)
"""

from luna import LunaNN

MODEL_NAME = "infinite_luna.pth"
EPOCHS = 10_000_000

def infinite_train() -> None:
    """ctrl-c to stop training Luna"""
    LunaNN(MODEL_NAME, cuda=True, verbose=True, epochs=EPOCHS)

if __name__ == "__main__":
    infinite_train()