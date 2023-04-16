"""
    Zero Knowledge Chess Engine

    by lipeeeee
"""

from luna_board import LunaBoard
from luna import Luna
import sys

def main() -> int:
    """Entry Point"""
    luna_chess = Luna()
    board = luna_chess.board
    moves = board.generate_legal_moves()
    for move in moves:
        print(move)

    return 0


if __name__ == "__main__":
    sys.exit(main())