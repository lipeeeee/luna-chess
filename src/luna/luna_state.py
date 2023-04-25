"""
    Luna-chess board state
"""

import chess
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

    # There is an infinite room for improvement...
    @staticmethod
    def serialize_board(board: chess.Board):
        """Exploration of new serialization techniques
            input_shape: (24, 8, 8)
            
            - pieces bitmap for each color(+6*2) = 12
            - material count for each color(+2) = 14
            - attacking squares for each color(+2) = 16
            - queen and kingside castling rights for each color(+4) = 20  
            - turn(+1) = 21
            - mat diff(+1) = 22 
            - move count(+1) = 23
            - en passant square(+1) = 24
            #TODO: move stack?
        """
        assert board.is_valid()

        # Define piece matrixes
        white_pawn_structure = np.zeros(64, np.uint8)
        white_bishop_structure = np.zeros(64, np.uint8)
        white_knight_structure = np.zeros(64, np.uint8)
        white_rook_structure = np.zeros(64, np.uint8)
        white_queen_structure = np.zeros(64, np.uint8)
        white_king_structure = np.zeros(64, np.uint8)
        white_attack_structure = np.zeros(64, np.uint8)
        white_mat_count = 0

        black_pawn_structure = np.zeros(64, np.uint8)
        black_bishop_structure = np.zeros(64, np.uint8)
        black_knight_structure = np.zeros(64, np.uint8)
        black_rook_structure = np.zeros(64, np.uint8)
        black_queen_structure = np.zeros(64, np.uint8)
        black_king_structure = np.zeros(64, np.uint8)
        black_attack_structure = np.zeros(64, np.uint8)
        black_mat_count = 0

        # board loop
        for i in range(64):
            pp = board.piece_at(i)

            if pp is not None:    
                #white
                if pp.symbol() == 'P':
                    white_pawn_structure[i] = 1
                    white_mat_count += 1
                elif pp.symbol() == 'B':
                    white_bishop_structure[i] = 1
                    white_mat_count += 3
                elif pp.symbol() == 'N':
                    white_knight_structure[i] = 1
                    white_mat_count += 3
                elif pp.symbol() == "R":
                    white_rook_structure[i] = 1
                    white_mat_count += 5
                elif pp.symbol() == 'Q':
                    white_queen_structure[i] = 1
                    white_mat_count += 9
                elif pp.symbol() == 'K':
                    white_king_structure[i] = 1
                #black
                elif pp.symbol() == 'p':
                    black_pawn_structure[i] = 1
                    black_mat_count += 1
                elif pp.symbol() == 'b':
                    black_bishop_structure[i] = 1
                    black_mat_count += 3
                elif pp.symbol() == 'n':
                    black_knight_structure[i] = 1
                    black_mat_count += 3
                elif pp.symbol() == 'r':
                    black_rook_structure[i] = 1
                    black_mat_count += 5
                elif pp.symbol() == 'q':
                    black_queen_structure[i] = 1
                    black_mat_count += 9
                elif pp.symbol() == 'k':
                    black_king_structure[i] = 1
                
            # attacking matrices
            if board.is_attacked_by(chess.WHITE, i):
                white_attack_structure[i] = 1
            elif board.is_attacked_by(chess.BLACK, i):
                black_attack_structure[i] = 1

        # reshape matrices
        white_pawn_structure = white_pawn_structure.reshape(8, 8)
        white_bishop_structure = white_bishop_structure.reshape(8, 8)
        white_knight_structure = white_knight_structure.reshape(8, 8)
        white_rook_structure = white_rook_structure.reshape(8, 8)
        white_queen_structure = white_queen_structure.reshape(8, 8)
        white_king_structure = white_king_structure.reshape(8, 8)
        white_attack_structure = white_attack_structure.reshape(8, 8)

        black_pawn_structure = black_pawn_structure.reshape(8, 8)
        black_bishop_structure = black_bishop_structure.reshape(8, 8)
        black_knight_structure = black_knight_structure.reshape(8, 8)
        black_rook_structure = black_rook_structure.reshape(8, 8)
        black_queen_structure = black_queen_structure.reshape(8, 8)
        black_king_structure = black_king_structure.reshape(8, 8)
        black_attack_structure = black_attack_structure.reshape(8, 8)

        # mix them all up        
        state = np.zeros((24, 8, 8), np.uint8)
        
        # turn 
        state[0] = (board.turn*1)
            
        # White positional features
        state[1] = white_pawn_structure
        state[2] = white_bishop_structure
        state[3] = white_knight_structure
        state[4] = white_rook_structure
        state[5] = white_queen_structure
        state[6] = white_king_structure
        # aditional white features
        state[7] = white_attack_structure
        state[8] = (white_mat_count)
        state[9] = (board.has_kingside_castling_rights(chess.WHITE) * 1)
        state[10] = (board.has_queenside_castling_rights(chess.WHITE) * 1)

        # Black positional features
        state[11] = black_pawn_structure
        state[12] = black_bishop_structure
        state[13] = black_knight_structure
        state[14] = black_rook_structure
        state[15] = black_queen_structure
        state[16] = black_king_structure
        # aditional black features
        state[17] = black_attack_structure
        state[18] = (black_mat_count * 1)
        state[19] = (board.has_kingside_castling_rights(chess.BLACK) * 1)
        state[20] = (board.has_queenside_castling_rights(chess.BLACK) * 1)

        # raw mat count diff from -1 to 1
        if white_mat_count == black_mat_count:
            state[21] = 0
        else:
            state[21] = (white_mat_count - black_mat_count)
        
        # move count in ply
        state[22] = (board.ply())

        # en passant square in float
        if board.ep_square is None:
            state[23] = 0
        else:
            state[23] = board.ep_square 

        return state

    @staticmethod
    def first_serialize_board(board: chess.Board):
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
    # since the pawn structure might be too complex for the current NN architecture
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
