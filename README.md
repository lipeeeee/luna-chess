<div align="center">

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![pytorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
![License](https://img.shields.io/github/license/R3nzTheCodeGOD/R3nzSkin.svg?style=for-the-badge)

# Luna Chess
</div>
Luna-Chess is a single-thread single-GPU chess engine rated around 1850, It trains a chess engine through pure self-play without <i>any</i> human knowledge except the rules of the game.

![Sem título](https://user-images.githubusercontent.com/62669782/233196743-ed90f2c3-7e2d-4a42-a469-b344e99115a4.png)


<p>

<p>

## Deep Neural Network
The neural network **f0** is parameterised by **0** and takes input the state **s** of the board. It has two outputs: a continuous value/evaluation of the board state **vθ(s)∈[−1,1]** from the prespective of the current player, and a policy **pθ(s)** that is a probability vector over all possible actions.

When training the network, at the end of each game of self-play, the neural net is provided training examples of the form **(st, πt, zt)**. **πt** is an esitmate of the policy from state **st** and **zt∈{−1,1}** is the final outcome of the game from the perspective of the player at **st**.

The neural net is trained to minimise the following loss function:

![image](https://user-images.githubusercontent.com/62669782/236341753-92420ce8-1636-46f3-900f-0d2407d1c38e.png)


The idea is that, over time, the network will learn what states eventually lead to wins or losses.

### Architecture
```python
self.conv1 = nn.Conv3d(1, args.num_channels, 3, stride=1, padding=1)
        
## Hidden
self.conv2 = nn.Conv3d(args.num_channels, args.num_channels * 2, 3, stride=1, padding=1)
self.conv3 = nn.Conv3d(args.num_channels * 2, args.num_channels * 2, 3, stride=1)
self.conv4 = nn.Conv3d(args.num_channels * 2, args.num_channels * 2, 3, stride=1)
self.conv5 = nn.Conv3d(args.num_channels * 2, args.num_channels, 1, stride=1)

self.bn1 = nn.BatchNorm3d(args.num_channels)
self.bn2 = nn.BatchNorm3d(args.num_channels * 2)
self.bn3 = nn.BatchNorm3d(args.num_channels * 2)
self.bn4 = nn.BatchNorm3d(args.num_channels * 2)
self.bn5 = nn.BatchNorm3d(args.num_channels)

self.fc1 = nn.Linear(args.num_channels*(self.board_x-4)*(self.board_y-4)*(self.board_z-4), 1024) #4096 -> 1024
self.fc_bn1 = nn.BatchNorm1d(1024)

self.fc2 = nn.Linear(1024, 512)
self.fc_bn2 = nn.BatchNorm1d(512)

self.fc3 = nn.Linear(512, 512)
self.fc_bn3 = nn.BatchNorm1d(512)

# output p(st)
self.fc4 = nn.Linear(512, self.action_size)

# output vt
self.fc5 = nn.Linear(512, 1)
```

## Monte-Carlo Tree Search for Policy Improvement
Given a state **s**, the neural network provides an estimate of the policy **pθ**. During the training phase, we wish to improve these estimates. This is accomplished using a Monte Carlo Tree Search (MCTS). In the search tree, each node represents a board configuration. A directed edge exists between two nodes i→j if a valid action can cause state transition from state **i** to **j**. Starting with an empty search tree, we expand the search tree one node (state) at a time. When a new node is encountered, instead of performing a rollout, the value of the new node is obtained from the neural network itself. This value is propagated up the search path.



![image](https://user-images.githubusercontent.com/62669782/236342095-b49c1869-f7b6-4f69-9a26-0c9f3be73412.png)

For the tree search, we maintain the following:
- **Q(s,a)**: the expected reward for taking action **a** from state **s**, i.e. the Q values;
- **N(s,a)**: the number of times we took action **a** from state **s** across simulations;
- **P(s,.)=pθ(s)**: the initial estimate of taking an action from the state s according to the policy returned by the current neural network;


From these, we can calculate **U(s,a)**, the upper confidence bound on the Q-values as


![image](https://user-images.githubusercontent.com/62669782/236342522-1df81c3c-17b4-4a0c-8bd3-ee452e11bdb3.png)


**cpuct** is a hyperparameter that controls the degree of exploration. To use MCTS to improve the initial policy returned by the current neural network, we initialise our empty search tree with **s** as the root. A single simulation proceeds as follows. We compute the action **a** that maximises the upper confidence bound **U(s,a)**. If the next state **s′** (obtained by playing action **a** on state **s**) exists in our tree, we recursively call the search on **s′**. If it does not exist, we add the new state to our tree and initialise **P(s′,.)=pθ(s′)** and the value **v(s′)=vθ(s′)** from the neural network, and initialise **Q(s′,a)** and **N(s′,a)** to 0 for all **a**. Instead of performing a rollout, we then propagate **v(s′)** up along the path seen in the current simulation and update all **Q(s,a)** values. On the other hand, if we encounter a terminal state, we propagate the actual reward (+1 if player wins, else -1).

After a few simulations, the **N(s,a)** values at the root provide a better approximation for the policy. The improved stochastic policy **π(s)** is simply the normalised counts **N(s,⋅)/∑b(N(s,b))**. During self-play, we perform MCTS and pick a move by sampling a move from the improved policy **π(s)**.

## Policy Iteration through Self-Play
(A High Level overview on how the network learns)


We initialise our neural network with random weights, thus starting with a random policy and value network. In each iteration of our algorithm, we play a number of games of self-play. In each turn of a game, we perform a fixed number of MCTS simulations starting from the current state **st**. We pick a move by sampling from the improved policy **πt**. This gives us a training example **(st,πt, _)**. The reward **_** is filled in at the end of the game: +1 if the current player eventually wins the game, else -1. The search tree is preserved during a game.


At the end of the iteration, the neural network is trained with the obtained training examples. The old and the new networks are pit against each other. If the new network wins more than a set threshold fraction of games, the network is updated to the new network. Otherwise, we conduct another iteration to augment the training examples.


## Feature engineering(ONLY USED IN SUPERVISED LEARNING WHICH IS DEPRECATED)
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

*a project by lipeeeee*
