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
This was by far the most annoying process in the building of the neural network architecture, from the board serialization data to type of data...

### Board Serialization
My approach to the board serializaton/encoding was to make it have every feature it could need about the board while not making it too complex, mainly because i didn't have the GPU or disk hardware to process complex datasets, the board was encoded in different ways, having a final shape of <b>(24, 8, 8)</b>

| Feature | Description | Shape |
| --- | --- | --- |
| Piece Bitmaps | Bitmaps of pices from the different colors, each piece for each color(6*2=12) | (12, 8, 8) |
| Turn | Binary feature to indicate who is playing next | (1, 8, 8) |
| Material | Integer value for the chess piece relative value count for each player | (2, 8, 8) |
| Material Difference | Material difference(`White.Material - Black.Material`) | (1, 8, 8) |
| En-Passant Square | Integer value for the en-passant square from 1-64, 0 if None | (1, 8, 8) |
| Attacking Squares | Bitmap of attacking squares for each color | (2, 8, 8) |
| Castling Rights | Binary features for each castling right option(kingside and queenside) for each player | (4, 8, 8) |
| PLY | Ply move count(half-moves) | (1, 8, 8)

### Float32 vs Int8
Choosing the datatype of the board serialization values played a big part in saving RAM, GPU computations and disk space.
A 2.5M dataset in `float32` would take 15GiB of RAM and disk space.
While a 5M dataset in `uint8` would take around 5GiB of RAM and disk space.

(I only had 16GiB to work with)

`float32` also brought problems such as the network not understanding the pieces bitmaps, because they were float values the network was thinking of the pieces as raw values instead of classes.

## Luna vs Stockfish
To test the efficacy of Luna's evaluation network I made a few functions to compare it against stockfish:

- A
- B

Note that if you ever want to build your own Luna model and compare it to stockfish you will have to download the [stockfish binaries](https://stockfishchess.org/download/).

## Self-Play
I also implemented a feature to Luna that allows her to play with itself 
    

![image](https://user-images.githubusercontent.com/62669782/233199778-5984d311-73ae-4a27-92c3-d291fdffd3ca.png)


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
- Stockfish related functions
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

## Luna Usage
A few examples of how Luna can be used.
```python
# Importing
import Luna

# Initializing the engine
luna_chess = Luna(verbose=True) # Verbose is advised since it outputs alot of info about what luna is doing(generating dataset, training, etc..) 
```

### Evaluating position
```python
# Initialize custom Luna board state(board with starting FEN)
luna_state = LunaState()
print(luna_state.board) # Cmd-Based visual representation of the chess board

# Integer Evaluation of the board, based around stockfish's eval function
evaluation = luna_chess.luna_eval(luna_state)
print(evaluation)

# If you want you can check the evaluation of random board states like such:
for i in range(1000):
  # Get random board
  random_board = luna_engine.random_board(max_depth=200)
  
  # LunaState with that board
  luna_state_temp = LunaState(board=random_board)
  
  # Evaluate random board
  evaluation_temp = luna_chess.luna_eval(luna_state_temp)
  print(f"{board}\nWITH EVAL:{evaluation_temp}\n")
```

### Evaluate moves
Evaluates moves using the neural network's trained evaluation function and the alpha beta pruning search algorithm. On a board, it will give us an evaluation on how good each legal move is after `luna_constants.SEARCH_DEPTH` moves.

```python

# Initialize custom Luna board state(board with starting FEN)
luna_state = LunaState()
print(luna_state.board) # Cmd-Based visual representation of the chess board

# Get the evaluation of the board after each legal_move with depth of luna_constants.SEARCH_DEPTH
eval_move_list = luna_chess.explore_leaves(luna_state)

# Sorting the list according to the current player
moves = sorted(eval_move_list, key=lambda x: x[0], reverse=luna_state.board.turn)
        
# Get best move from sorted moves
best_move = move[0][1]

# Top 3 moves
print("Calculated Top 3:")        
for i,m in enumerate(move[0:3]):
    print(f"  {m}")
```

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


Material count: The total value of all the pieces on the board.

Piece mobility: The number of squares each piece can move to on the board.

King safety: How exposed the king is to threats, such as the number of attacking pieces and the number of squares the king can move to.

Pawn structure: The arrangement of the pawns on the board, including isolated pawns, doubled pawns, and pawn chains.

Control of the center: The number of pieces that control the central squares of the board.

Development: The number of pieces that have been moved from their starting position.

Space control: The number of squares controlled by each player.

Tempo: The number of moves a player has made compared to their opponent.

Piece placement: The location of each piece on the board and how well it is placed for potential future moves.

Threats: The number of threats each player has on the board.

1. implement Luna in a webserver(such as firebase)
2. Add a bit of randomness when it comes to computer moves
3. combine Alpha beta pruning w, transposition tables, quiescence search, and iterative deepening 
