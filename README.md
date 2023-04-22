<div align="center">

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![pytorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
![License](https://img.shields.io/github/license/R3nzTheCodeGOD/R3nzSkin.svg?style=for-the-badge)

# Luna Chess
</div>
<b>Luna-Chess</b> is a chess engine rated around <b>(TBD)</b>, It works using a <b>Deep Neural Network</b> to evaluate a board state and then using <b>Alpha–beta pruning</b> to search through the space of possible future board states and evaluate what is the best move over-time.


![Sem título](https://user-images.githubusercontent.com/62669782/233196743-ed90f2c3-7e2d-4a42-a469-b344e99115a4.png)


<p>

<p>

## Deep Neural Network
I used pytorch because of it's explicit control over networks(the architecture can be improved by addding pooling2d and other types of layers), the simplified goal of this network is to take in a board state in ``a1()`` and return a evaluation score in ``last()``:
```python
LunaNN(
  (a1): Conv2d(5, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (a2): Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (a3): Conv2d(16, 32, kernel_size=(3, 3), stride=(2, 2))
  (b1): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (b2): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (b3): Conv2d(32, 64, kernel_size=(3, 3), stride=(2, 2))
  (c1): Conv2d(64, 64, kernel_size=(2, 2), stride=(1, 1), padding=(1, 1))
  (c2): Conv2d(64, 64, kernel_size=(2, 2), stride=(1, 1), padding=(1, 1))
  (c3): Conv2d(64, 128, kernel_size=(2, 2), stride=(2, 2))
  (d1): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1))
  (d2): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1))
  (d3): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1))
  (last): Linear(in_features=128, out_features=1, bias=True)
  (loss): MSELoss()
)
```

## Feature engineering
...


## Board Serialization
...

## Luna vs Stockfish
...

## Project Architecture
I aimed to create a deep learning model that could **easily** be used as a package, so I conceptualized this project into an object-oriented approach, making it so that by just doing this:
```python
import luna
```
You have acess to:
- Luna neural network
- Luna evaluation function
- Luna custom board states
- Luna dataset creation and handling
- All constants used by Luna
- The actual engine logic, obviously

### The architecture:
```
Wrapper(either html or anything else) ->
    Luna ->
        Luna_State ->
        Luna_Eval ->
            Luna_NN ->
            Luna_dataset ->
```

## Evaluating position
```python
import luna

luna_engine = Luna()
board = chess.Board()

evaluation = luna_engine.luna_eval(board)
```

## Self-Play
I also implemented a feature to Luna that allows her to play with itself 
  
  
  ![image](https://user-images.githubusercontent.com/62669782/233199778-5984d311-73ae-4a27-92c3-d291fdffd3ca.png)

  
  
## HTML Wrapper
To test the usablity of the Luna package I made a VERY SIMPLE **HTML web server wrapper**, that just uses Luna as backend logic while HTML is used to display Luna's contents.

You can check the wrapper at ``src/luna_html_wrapper.py``.

You can also(on the project main folder) run the web server with:
```makefile
make web
```

# Usage
```
# install every package
pip install -r requirements.txt
# run web server
make web
```

TODO
------

stockfish 0 depth dataset~
 better serialize, even ebtter!!!
 Legal moves: The set of legal moves that are available to the player at the current board state.

History of moves: A sequence of previous moves made in the game, along with their corresponding board states.

Material balance: The difference in the number and value of pieces captured by each player.

Positional features: Features that describe the position of each piece, such as the number of attackers and defenders for each piece, the control of key squares on the board, and pawn structure.

Opening book: The history of moves can also be used to create an opening book, which is a collection of known openings and their corresponding moves. By using an opening book as a reference, the neural network can quickly identify strong opening moves and avoid making weak ones.

Overall, providing the history of moves as input to a neural network-based chess engine can help the neural network to develop a more sophisticated understanding of the game and make better decisions based on a broader range of information.

the player to move, castling rights, and en passant possibility.

improve pgn quality

Castling rights: This is a binary feature that indicates whether each player can still castle on either side of the board.

En passant: This is also a binary feature that indicates whether a pawn can currently be captured en passant.

Move count: This is an integer feature that keeps track of the total number of moves made in the game so far.

Piece count: This is a set of integer features that indicates the number of pieces of each type that each player has left on the board.

Material count: This is a set of integer features that indicates the total value of each player's pieces on the board, based on standard chess valuations.

Threats: This is a set of binary features that indicates which squares on the board are currently being threatened by each player's pieces.

Mobility: This is a set of integer features that indicates the number of legal moves that each player can make on their turn.

1. implement Luna in a webserver(such as firebase)
2. Add a bit of randomness when it comes to computer moves
3. combine Alpha beta pruning w, transposition tables, quiescence search, and iterative deepening 