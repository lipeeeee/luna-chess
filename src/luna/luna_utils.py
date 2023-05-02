"""
    Util Classes/functions for luna to prevent circular imports
"""

import chess.engine

sf: chess.engine.SimpleEngine = None

class AverageMeter(object):
    """From https://github.com/pytorch/examples/blob/master/imagenet/main.py"""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def __repr__(self):
        return f'{self.avg:.2e}'

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class dotdict(dict):
    """DotDict for Network HyperParams"""

    def __getattr__(self, name):
        return self[name]

def init_stockfish() -> None:
    """Initializes stockfish"""
    global sf 
    sf = chess.engine.SimpleEngine.popen_uci('./content/stockfish.exe')

def close_stockfish() -> None:
    """Closes stockfish process"""
    # cant find close process

def stockfish(board:chess.Board, depth) -> int:
    """Stockfish evaluator"""
    assert sf is not None

    result = sf.analyse(board, chess.engine.Limit(depth=depth))
    score = result['score'].white().score()
    
    #if score == None:
    #    print("none.." + board.fen())
    #    raise Exception("none stockfish value")
    
    return score