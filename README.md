<div align="center">

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![Keras](https://img.shields.io/badge/Keras-%23D00000.svg?style=for-the-badge&logo=Keras&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
![pytorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![tensorflow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
<!-- [![Windows](https://img.shields.io/badge/Platform-Windows-0078d7.svg?style=for-the-badge)](https://en.wikipedia.org/wiki/Microsoft_Windows) -->
<!-- [![License](https://img.shields.io/github/license/R3nzTheCodeGOD/R3nzSkin.svg?style=for-the-badge)](LICENSE) -->

# Luna-Chess
</div>
<b>Luna-Chess</b> is a chess engine rated around <b>(TBD)</b>, I built it with little to no knowledge about the backend of chess engines(CE), I conceptualized it by reading about Reinforcement Learning and Deep Learning and it's applicability to chess.

<p>

Ultimately i chose a <b>Deep Neural Network</b> approach since reinforcement learning would be much more computationaly expensive and also a neural network would adapt much better to new states since it could detect patterns with ease such as <i>zugzwang</i> and <i> perpetual checks</i>.
<p>

## Architecture
```python
optimizer = 'adam'
loss = 'mean_squared_error'
N = 864
input_shape = (8, 8, 12)

model = keras.models.Sequential([
    layers.Conv2D(64, 3, activation='relu', input_shape=input_shape),
    layers.Conv2D(64, 3, activation='relu'),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dense(1, activation='linear')
])

model.compile(optimizer=optimizer, loss=loss)
```


TODO
------

1. implement Luna in a webserver(such as firebase)
2. Add a bit of randomness when it comes to computer moves
3. make so winning side doesnt play random moves just because it is winnign