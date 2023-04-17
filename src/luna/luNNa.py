"""
    Luna neural net
"""

from torch import *

class luNNa(nn.Module):
    """Pytorch Neural Network"""

    def __init__(self, model_path) -> None:
        self.model_path = model_path

    def define(self, input_shape) -> None:
        """Define neural net"""

    def load(self, path) -> None:
        """Load luna from a .pt file"""

    def save(self, path) -> None:
        """Save luna weights and biases and everything else into a .pt file"""

    def train(self, N=0) -> None:
        """Train Luna on a pgn file directory(on N games)"""
        assert not self.model_exists()
    
        return
    
    def model_exists(self) -> bool:
        """Checks if there is a pre-saved model"""
        return os.path.exists(self.model_path)