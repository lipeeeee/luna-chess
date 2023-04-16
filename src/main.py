"""
    Zero Knowledge Chess Engine

    by lipeeeee
"""

import chess as chess
import sys

def main() -> int:
    """Entry Point"""
    board = chess.Board()
    moves = board.generate_legal_moves()
    for move in moves:
        print(move)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())