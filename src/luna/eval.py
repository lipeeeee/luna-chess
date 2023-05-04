"""
    Hand-Made Luna evaluator
    made to test and debug new models and check if it overfit
"""

import sys
from luna import Luna

VERBOSE = True
NUM_EVALS = 2000 # keeping it over 1000 results in better eval

def eval() -> int:
    """Hand-Made Eval"""

    # Start luna for evaluation
    luna_chess = Luna(verbose=VERBOSE)    
    
    # Difference between luna's prediction and stockfish's prediction (A - B)
    luna_chess.luna_stockfish_diff(NUM_EVALS)

    # Printing of various random board states, luna vs stockfish
    luna_chess.print_luna_vs_stockfish(int(NUM_EVALS / 2))
    # callback link to link deque 1 link
    return 0

if __name__ == "__main__":
    if VERBOSE: print(f"[EVAL] Running eval.py to self-evaluate model on random states")
    sys.exit(eval())