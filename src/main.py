"""
    Luna-Chess, a chess engine rated around X

    Project Architecture    
    Wrapper(either html or anything else) ->
        Luna ->
            Luna_Utils ->
            Luna_State ->
            Luna_Eval ->
                Luna_NN ->
                Luna_dataset ->

    by lipeeeee
"""

from luna import *
import random
import chess
import sys

def main() -> int:
    """Entry Point"""
    luna = Luna(True)
    luna.print_luna_vs_stockfish(600)

    return 0



if __name__ == "__main__":
    sys.exit(main())