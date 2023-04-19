"""
    Luna-chess board state
"""

import chess
from chess import Move
import numpy as np

class LunaState(): 
    """Luna-chess custom board state"""

    def __init__(self, board:chess.Board=None):
        if board is None:
            self.board = chess.Board()
        else:
            self.board = board

    def key(self):
        """State key"""
        return (self.board.board_fen(), self.board.turn, self.board.castling_rights, self.board.ep_square)
    
    def state_after_move(self, move:Move):
        """Returns a board copy after a move"""
        return self.serialize_board

    @staticmethod
    def serialize_board(board: chess.Board):
        """Serialize a chess board into a NN readable format
            1. Encode board
            2. Encode pawn structure(-1 for black 1 for white 0 for none)
            3. Encode board into binary representation(4bit)
            4. Encode turn
        """
    
        # Check if valid board before preprocessing
        assert board.is_valid()

        board_state = np.zeros(64, np.uint8)
        pawn_structure = np.zeros(64, np.int8)
        for i in range(64):
            pp = board.piece_at(i)
            
            if pp is None:
                continue

            # 1. Board state encoding
            board_state[i] = {"P": 1, "N": 2, "B": 3, "R": 4, "Q": 5, "K": 6, \
                    "p": 9, "n":10, "b":11, "r":12, "q":13, "k": 14}[pp.symbol()]
            
            # 2. Encode pawn structure            
            if pp.symbol() == "P" or pp.symbol() == "p":
                pawn_structure[i] = {"P": 1, "p": -1}[pp.symbol()]

        board_state = board_state.reshape(8, 8) 
        pawn_structure = pawn_structure.reshape(8, 8)

        state = np.zeros((6, 8, 8), np.uint8)

        # 2. Pawn structure
        state[0] = pawn_structure

        # 3. Binary board state
        state[1] = (board_state>>3)&1
        state[2] = (board_state>>2)&1
        state[3] = (board_state>>1)&1
        state[4] = (board_state>>0)&1

        # 4. 5th column is who's turn it is
        state[5] = (board.turn*1.0)

        return state

    @staticmethod
    #since the pawn structure might be too complex for the current NN architecture
    # this is a version without the encoding of pawn struct
    def no_pawn_serialize_board(board: chess.Board):
        """Serialize a chess board into a NN readable format
            1. Encode board
            2. Encode board into binary representation(4bit)
            3. Encode turn
        """
    
        # Check if valid board before preprocessing
        assert board.is_valid()

        board_state = np.zeros(64, np.uint8)
        for i in range(64):
            pp = board.piece_at(i)
            
            if pp is None:
                continue

            # 1. Board state encoding
            board_state[i] = {"P": 1, "N": 2, "B": 3, "R": 4, "Q": 5, "K": 6, \
                    "p": 9, "n":10, "b":11, "r":12, "q":13, "k": 14}[pp.symbol()]
            
        # additional processing
        if board.has_queenside_castling_rights(chess.WHITE):
            assert board_state[0] == 4
            board_state[0] = 7
        if board.has_kingside_castling_rights(chess.WHITE):
            assert board_state[7] == 4
            board_state[7] = 7
        if board.has_queenside_castling_rights(chess.BLACK):
            assert board_state[56] == 8+4
            board_state[56] = 8+7
        if board.has_kingside_castling_rights(chess.BLACK):
            assert board_state[63] == 8+4
            board_state[63] = 8+7
        if board.ep_square is not None:
            assert board_state[board.ep_square] == 0
            board_state[board.ep_square] = 8
    
        board_state = board_state.reshape(8, 8) 
        
        state = np.zeros((5, 8, 8), np.uint8)

        # 2. Binary board state
        state[0] = (board_state>>3)&1
        state[1] = (board_state>>2)&1
        state[2] = (board_state>>1)&1
        state[3] = (board_state>>0)&1

        # 3. 4th column is who's turn it is
        state[4] = (board.turn*1.0)

        return state

    def edges(self):
        """Self-Explanatory..."""
        return list(self.board.legal_moves)
