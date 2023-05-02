"""
    python-chess luna wrapper
"""

from __future__ import print_function
import numpy as np
import chess

# Helper Functions
def to_np(board: chess.Board):
    a = [0] * (8*8*6)
    for sq, pc in board.piece_map().items():
        a[sq * 6 + pc.piece_type - 1] = 1 if pc.color else -1
    return np.array(a)

def from_move(move: chess.Move):
    return move.from_square*64+move.to_square

def to_move(action):
    to_sq = action % 64
    from_sq = int(action / 64)
    return chess.Move(from_sq, to_sq)

def who(turn: bool):
  """Who is playing, 1 for white -1 for black"""
  # return int(turn)
  return 1 if turn else -1

def mirror_move(move: chess.Move):
  return chess.Move(chess.square_mirror(move.from_square), chess.square_mirror(move.to_square))

# Game Outcomes
CHECKMATE = 1
STALEMATE = 2
INSUFFICIENT_MATERIAL = 3
SEVENTYFIVE_MOVES = 4
FIVEFOLD_REPETITION = 5
FIFTY_MOVES = 6
THREEFOLD_REPETITION = 7

class ChessGame():
    """python-chess wrapper"""

    def __init__(self):
        super(ChessGame, self).__init__()

    def getInitBoard(self):
        """Initial Board State example"""
        return chess.Board()

    def getBoardSize(self):
        """Board Dimensions"""
        # (a,b) tuple
        # 6 piece type
        return (8, 8, 6)

    def toArray(self, board):
        """Serialized board"""
        return to_np(board)

    def getActionSize(self):
        """Number of actions possible"""
        return 64*64
        # return self.n*self.n*16+1

    def getNextState(self, board: chess.Board, player: chess.Color, action):
        """Get next state given board and action"""
        # if player takes action on board, return next (board,player)
        # action must be a valid move
        
        assert(who(board.turn) == player)
        move = to_move(action)
        if not board.turn:
            # assume the move comes from the canonical board...
            move = mirror_move(move)
        if move not in board.legal_moves:
            # could be a pawn promotion, which has an extra letter in UCI format
            move = chess.Move.from_uci(move.uci()+'q') # assume promotion to queen
            if move not in board.legal_moves:
                assert False, "%s not in %s" % (str(move), str(list(board.legal_moves)))
        board = board.copy()
        board.push(move)
        return (board, who(board.turn))

    def getValidMoves(self, board: chess.Board, player: chess.Color):
        """Fixed size binary vector"""
        assert(who(board.turn) == player)
        
        acts = [0] * self.getActionSize()
        for move in board.legal_moves:
          acts[from_move(move)] = 1
        
        return np.array(acts)

    def getGameEnded(self, board: chess.Board) -> float:
        """return 0 if not ended, 1 if player 1 won, -1 if player 1 lost"""
        
        outcome = board.outcome()
        reward = 0.0
        if outcome is not None:
            if outcome.winner is None:
                # draw, very little reward value
                reward = 1e-4
            else:
                if outcome.winner == board.turn:
                    reward = 1.0
                else:
                    reward = -1.0
        
        return reward

    def getCanonicalForm(self, board: chess.Board, player: chess.Color):
        """return state if player==1, else return -state if player==-1"""
        assert(who(board.turn) == player)

        if board.turn:
            return board
        else:
            return board.mirror()

    def getSymmetries(self, board, pi):
        return [(board,pi)]

    def stringRepresentation(self, board: chess.Board):
        return board.fen()

    @staticmethod
    def display(board):
        print(board)
