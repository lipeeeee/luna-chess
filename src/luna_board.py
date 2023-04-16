"""
    Custom python-chess.Board for Luna
"""

from chess import *
import numpy as np

class LunaBoard(Board):
    """Custom Implementation of python-chess.Board"""

    def __init__(self) -> None:
        super().__init__()

    def serialize(self) -> np.array:
        """Serialize:
            1. FEN STRING(split)
            2. PIECE MOBILITY(Compute the number of legal moves available to each pawn, knight, bishop, rook, queen, and king)
            3. PAWN STRCUTURE(number of pawns on each file and rank, the presence of pawn chains or isolated pawns, and the presence of passed pawns. )
            4. MOVE COUNT?
        """

