"""
    Chess Players

    TODO: Make players Inherit from parent class `Player`
"""

import chess
import random
import numpy as np
from .luna_game import ChessGame, who, from_move, mirror_move
from stockfish import Stockfish

def move_from_uci(board: chess.Board, uci: str) -> chess.Move | None:
    """Checks if valid move"""
    try:
        move = chess.Move.from_uci(uci)
    except ValueError:
        print('expected an UCI move')
        return None
    if move not in board.legal_moves:
        print('expected a valid move')
        return None
    return move

class RandomPlayer(object):
    """Random player, always plays random moves"""
    # Environment
    game: ChessGame
    
    def __init__(self, game: ChessGame) -> None:
        super(RandomPlayer, self).__init__()
        self.game = game

    def play(self, board) -> chess.Move:
        """Play random moves"""
        valids = self.game.getValidMoves(board, who(board.turn))
        moves = np.argwhere(valids == 1)
        return random.choice(moves)[0]

class HumanChessPlayer(object):
    """Human Player"""

    def __init__(self) -> None:
        super(HumanChessPlayer, self).__init__()

    def play(self, board) -> chess.Move:
        """Ask for Input and return move"""
        mboard = board
        if board.turn:
            mboard = board.mirror()
        
        print('Valid Moves', end=':')
        for move in mboard.legal_moves:
            print(move.uci(), end=',')
        
        # Input
        print()
        human_move = input()
        
        # Check Move
        move = move_from_uci(mboard, human_move.strip())
        if move is None:
            print('try again, e.g., %s' % random.choice(list(mboard.legal_moves)).uci())
            return self.play(board)
        
        if board.turn:
            move = mirror_move(move)
        return from_move(move)

class StockFishPlayer(object):
    """Stockfish"""
    
    # Actual Engine
    stockfish: Stockfish

    # Engine Elo
    elo: int
    
    # SkillLevel param (0-20)
    skill_level: int

    # Search Depth
    depth: int

    # Max think time
    think_time: int

    def __init__(self, elo=1000, skill_level=10, depth=10, think_time=30):
        """
            elo does not matter
            only skilllevel, depth is the main factor affect winrate
        """
        super(StockFishPlayer, self).__init__()

        self.stockfish = Stockfish(parameters={"Threads": 2, "Minimum Thinking Time": think_time})
        self.stockfish.set_elo_rating(elo)
        self.stockfish.set_skill_level(skill_level)
        self.stockfish.set_depth(depth)

    def play(self, board) -> int:
        """Get move from stockfish given board"""
        self.stockfish.set_fen_position(board.fen())
        uci_move = self.stockfish.get_best_move()
        move = move_from_uci(board, uci_move.strip())
        return from_move(move)
