"""
    Luna-Chess
"""

class Luna():
    """Luna chess engine main class"""

    def __init__(self) -> None:
        self.optimizer = 'Adam'
        self.loss = 'categorical_crossentropy'

    def train(self, folder, notation="pgn") -> None:
        """Train Luna by reading chess games of a folder, with the pgn notation by default"""        
        ...