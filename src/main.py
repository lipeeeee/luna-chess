"""
    Zero Knowledge Chess Engine

    Luna ->
        Luna_NN ->
            Luna_Eval ->
        Luna_dataset ->
        Luna_Eval ->

    by lipeeeee
"""

from luna import Luna
import sys

def main() -> int:
    """Entry Point"""
    luna_chess = Luna(verbose=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())