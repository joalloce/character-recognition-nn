# Character Recognition using Neural Network 

Neural Network the recognizes the first 10 capital consonant letters of the alphabet.
The letters are B, C, D, F, G, H, J, K L and M.

Yes, the neural network is from scratch in Python.

It uses the following technologies:

- Python
- NumPy
- Jupyter Notebook

## Setup
To train a new neural network (It could last long depending on the number of iterations):
```sh
python main.py start
```
To continue the neural network training. Training started needed. (W1.txt, W2.txt, numIterations.txt in tmp/ folder):
```sh
python main.py start
```
To test the neural network. Training started needed. (W1.txt, W2.txt in tmp/ folder). Results are getting better after 50 iterations or so.
```sh
python main.py test
```
When you start the training. The process could last long. You can stop it by pressing `Crtl + C`. You can resume it by using the command `python main.py continue`.

You can run the `python main.py test` command by creating a <i>tmp/</i> folder and use a set of W1.txt and W2.txt in the training folder. For the `python main.py continue` command you need the numIterations.txt with a number in the first line of the file. For instance, if you want to continue a training, you can take my try1 W1.txt and W2.txt files and paste in the tmp/ folder. Create a numIterations.txt file with the number <i>50</i>. (In my try1, the number of iterations was 50). Training might not make the neural network to perform better.