"""
    Luna neural net
"""

import os
from torch import *
from luna import LUNA_MAIN_FOLDER, TRAINING_FOLDER, CURRENT_MODEL

class LunaNN(nn.Module):
    """Pytorch Neural Network"""

    def __init__(self, model_file=CURRENT_MODEL) -> None:
        self.model_file = model_file # .pt file(ex: main_luna.pt)
        self.model_path = os.path.join(LUNA_MAIN_FOLDER, TRAINING_FOLDER, model_file)

    def define(self, input_shape) -> None:
        """Define neural net"""

    def load(self) -> None:
        """Load luna from a .pt file"""

    def save(self) -> None:
        """Save luna weights and biases and everything else into a .pt file"""

    def train(self, N=0) -> None:
        """Train Luna on a pgn file directory(on N games)"""
        assert not self.model_exists()
    
        return
    
    def model_exists(self) -> bool:
        """Checks if there is a pre-saved model"""
        return os.path.exists(self.model_path)