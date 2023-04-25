"""
    Util functions for luna to prevent circular imports
"""

import chess.engine

sf: chess.engine.SimpleEngine = None

def init_stockfish() -> None:
    """Initializes stockfish"""
    global sf 
    sf = chess.engine.SimpleEngine.popen_uci('./content/stockfish.exe')

def close_stockfish() -> None:
    """Closes stockfish process"""
    # cant find close process

def stockfish(board:chess.Board, depth) -> float:
    """Stockfish evaluator"""
    assert sf is not None

    result = sf.analyse(board, chess.engine.Limit(depth=depth))
    score = result['score'].white().score()
    
    #if score == None:
    #    print("none.." + board.fen())
    #    raise Exception("none stockfish value")
    
    return score