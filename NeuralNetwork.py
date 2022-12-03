import numpy as np
import utils
import fileManager

# starting weights ranges
max_weight = 0.3
min_weight = -0.3
weight_range = 0.6


class NeuralNetwork():
    '''
    Neural network
    default values:
     - iterations = 100
     - hidden layer nodes = 250
    '''

    def __init__(self, layers=[2500, 250, 10], iterations=100) -> None:
        self.iterations = iterations
        '''
        [0] input layer size
        [1] hidden layer size
        [2] output layer size
        '''
        self.layers = layers
        '''
        [b1] first bias array
        [W1] first weight array
        [b2] second bias array
        [W2] second weight array
        '''
        self.params = {}
        self.X = None  # input
        self.Y = None  # output
        self.error = 0
        self.numIterations = 0

    def initialize(self):
        '''
        Initialize all the layers at 0
        '''
        self.params["b1"] = np.zeros((self.layers[1]))
        self.params["W1"] = np.zeros((self.layers[0], self.layers[1]))
        self.params["b2"] = np.zeros((self.layers[2]))
        self.params["W2"] = np.zeros((self.layers[1], self.layers[2]))

    def init_weights_randomly(self):
        '''
        Initialize weights with random between -0.3 to 0.3. no zeros allowed.
        '''
        self.numIterations = 0  # zero
        self.error = 0  # zero

        for i in range(self.layers[0]):
            for j in range(self.layers[1]):
                number = (np.random.randint(low=0, high=600)/1000) - max_weight
                while number == 0.0:  # if random equals zero take another random
                    number = (np.random.randint(
                        low=0, high=600)/1000) - max_weight
                self.params["W1"][i][j] = number

        for i in range(self.layers[1]):
            for j in range(self.layers[2]):
                number = (np.random.randint(low=0, high=600)/1000) - max_weight
                while number == 0.0:  # if random equals zero take another random
                    number = (np.random.randint(
                        low=0, high=600)/1000) - max_weight
                self.params["W2"][i][j] = number

    def setWeights(self, W1, W2):
        self.params["W1"] = W1
        self.params["W2"] = W2

    def setNumIterations(self, numIterations):
        self.numIterations = numIterations

    def clear_weights(self):
        '''
        Set weights to zeros
        '''
        self.params["W1"] = np.zeros((self.layers[0], self.layers[1]))
        self.params["W2"] = np.zeros((self.layers[1], self.layers[2]))

    def compute_b2(self):
        '''
        Calculate the second bias
        '''
        for i in range(self.layers[2]):
            Z = self.compute_net_sum_b2(i)
            self.params["b2"][i] = self.sigmoid(Z)

    def compute_net_sum_b2(self, position):
        '''
        Net sum of second bias. Depends on the second weights and the first bias
        '''
        sum = 0.0
        for i in range(self.layers[1]):
            sum += self.params["b1"][i] * self.params["W2"][i][position]

        return sum

    def compute_b1(self):
        '''
        Calculate the first bias
        '''
        for i in range(self.layers[1]):
            Z = self.compute_net_sum_b1(i)
            self.params["b1"][i] = self.sigmoid(Z)

    def compute_net_sum_b1(self, position):
        '''
        Net sum of first bias. Depends on the first weights and the input
        '''
        sum = 0.0
        for i in range(self.layers[0]):
            sum += self.X[i] * self.params["W1"][i][position]

        return sum

    def sigmoid(self, Z):
        '''
        Activation function
        Sigmoid function
        '''
        return 1/(1+np.exp(-Z))

    def forward_propagation(self):
        '''
        Compute both first and second bias
        '''
        self.compute_b1()
        self.compute_b2()

    def compute_error(self):
        '''
        Calculate the error of each output
        '''
        self.error = 0
        for i in range(self.layers[2]):
            self.error += abs(self.Y[i] - self.params["b2"][i])

    def check_if_error_surpass_threshold(self, threshold):
        '''
        if error > threshold
        '''
        if (self.error/self.layers[2]) > threshold:
            return True

        return False

    def adjust_W1(self, alpha):
        '''
        Compute the first weights. Depends on second bias factor, alpha and input
        '''
        for i in range(self.layers[1]):
            sum = self.sum_b2_adjust_factor(i)
            b2_factor = self.params["b1"][i] * (1 - self.params["b1"][i]) * sum
            for j in range(self.layers[0]):
                self.params["W1"][j][i] += alpha * \
                    b2_factor * self.X[j]

    def sum_b2_adjust_factor(self, position):
        '''
        Sum the second bias factor. Depends on second weights
        '''
        sum = 0
        for i in range(self.layers[2]):
            sum += self.get_b2_adjust_factor(i) * \
                self.params["W2"][position][i]

        return sum

    def adjust_W2(self, alpha):
        '''
        Compute the second weights. Depends on second bias factor, alpha and first bias
        '''
        for i in range(self.layers[2]):
            b2_factor = self.get_b2_adjust_factor(i)
            for j in range(self.layers[1]):
                self.params["W2"][j][i] += alpha * \
                    b2_factor * self.params["b1"][j]

    def get_b2_adjust_factor(self, position):
        '''
        Calculate the second bias factor
        '''
        return (self.Y[position] - self.params["b2"][position]) * self.params["b2"][position] * (1 - self.params["b2"][position])

    def back_propagation(self, alpha):
        '''
        Adjust the first and the second weights
        '''
        self.adjust_W1(alpha)
        self.adjust_W2(alpha)

    def fit(self, train_X, train_Y, threshold, alpha):
        '''
        Fit method
        '''
        stop = False
        while stop == False and self.numIterations < self.iterations:
            self.numIterations += 1
            stop = True
            # train with all the samples
            for i in range(len(train_X)):
                print("Iteration", self.numIterations, "sample", i+1)
                # print(train_Y[i]) #DELETE
                # set the input and output
                self.X = train_X[i]
                self.Y = train_Y[i]

                self.forward_propagation()
                self.compute_error()
                # print(self.params["b2"]) #DELETE

                # perform an ajustment if error surpass threshold
                if self.check_if_error_surpass_threshold(threshold):
                    stop = False
                    self.back_propagation(alpha)

            self.save_data()  # save the data for every iteration

    def save_data(self):
        '''
        Save the weights and the number of iterations
        '''
        fileManager.save_weights(
            self.params["W1"], self.params["W2"], self.layers)
        fileManager.save_numIterations(self.numIterations)

    def predict(self, X):
        '''
        Make a prediction
        '''
        self.X = X
        self.forward_propagation()  # performs a foward propagation
        return self.params["b2"]  # output

    def accuracy(self, Y):
        '''
        Calculate the accuracy of the output and the expected Y
        '''
        sum = 0.0
        for i in range(self.layers[2]):
            sum += abs(Y[i] - self.params["b2"][i])
        return 1 - (sum/self.layers[2])

    def test(self, test_X, test_Y):
        '''
        Test all the test samples. Calculates metrics
        '''
        match_counter = 0  # number of tests correctly predicted
        sum = 0.0  # sum of accuracies
        length = len(test_X)
        for i in range(length):
            # print(test_Y[i]) #DELETE
            prediction_vector = self.predict(test_X[i])  # predict
            # print(np.max(prediction_vector), prediction_vector) #DELETE
            index = prediction_vector.tolist().index(np.max(prediction_vector))
            prediction = utils.get_letter_from_index(index)  # letter predicted
            index = test_Y[i].tolist().index(np.max(test_Y[i]))
            expected = utils.get_letter_from_index(index)  # letter expected
            if expected == prediction:
                match_counter += 1

            acc = self.accuracy(test_Y[i])  # calculate the accuracy
            sum += acc
            print("Test", i + 1, "Prediction:", prediction,
                  "Expected:", expected)

        print("Average Accuracy:", round((sum/length) * 100, 2), "%")
        print("Predicted", match_counter, "from", length,
              "Performance:", round((match_counter/length) * 100, 2), "%")
