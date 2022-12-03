import fileManager
import utils
import NeuralNetwork
import numpy as np
import sys

threshold = 0.1
alpha = 0.005


def op_continue():
    # load samples from test_samples/ and train_samples/
    train_X, train_Y, test_X, test_Y = utils.load()

    nn = NeuralNetwork.NeuralNetwork(layers=[2500, 250, 10], iterations=100)
    nn.initialize()

    # tmp/ folder needed with the files W1.txt, W2.txt and numIteration.txt
    W1, W2 = fileManager.get_weights()
    numIterations = fileManager.getIterations()

    nn.setWeights(W1, W2)
    nn.setNumIterations(numIterations)

    nn.fit(train_X, train_Y, threshold, alpha)

    nn.test(test_X, test_Y)


def op_start():
    # load samples from test_samples/ and train_samples/
    train_X, train_Y, test_X, test_Y = utils.load()

    nn = NeuralNetwork.NeuralNetwork(layers=[2500, 250, 10], iterations=100)
    nn.initialize()
    nn.init_weights_randomly()

    nn.fit(train_X, train_Y, threshold, alpha)

    nn.test(test_X, test_Y)


def op_test():
    # load samples from test_samples/ and train_samples/
    train_X, train_Y, test_X, test_Y = utils.load()

    nn = NeuralNetwork.NeuralNetwork(layers=[2500, 250, 10], iterations=100)
    nn.initialize()

    # tmp/ folder needed with the files W1.txt and W2.txt
    W1, W2 = fileManager.get_weights()
    nn.setWeights(W1, W2)

    nn.test(test_X, test_Y)


def main():
    if len(sys.argv) == 1:
        print("start: start a new training")
        print("continue: continue training from the last iteration done.")
        print("Require the weights on folder tmp/ and the numIteration.txt with the number of the last iteration.")
        print("test: test the NN. Require the weights on folder tmp/.")
    else:
        op = sys.argv[1]
        if op == "continue":
            op_continue()
        elif op == "start":
            op_start()
        elif op == "test":
            op_test()
        else:
            print("Please type start, continue or test.")
            print(op, "is not a valid op.")


if __name__ == '__main__':
    main()
