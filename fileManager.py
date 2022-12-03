import numpy as np
import glob
import os

# letters
letters = ["B", "C", "D", "F", "G", "H", "J", "K", "L", "M"]


def get_samples(origin):
    '''
    Get the samples from directory origin.
    It takes the letters from the dict letters
    Returns an array of samples.
    '''
    samples = []
    for letter in letters:
        letter_set = get_letter_samples(origin, letter)
        for sample in letter_set:
            samples.append(sample)

    samples = np.array(samples)  # to numpy array

    return samples


def get_letter_samples(origin, letter):
    '''
    Get samples from a letter
    The sample file names should start with the letter 
    returns the samples of a letter
    '''
    samples = []
    os.chdir(origin)
    for path in glob.glob(letter + "*"):
        matrix = np.genfromtxt(str(path), delimiter=" ", skip_footer=1)
        array = matrix.flatten()  # turns matrix to a flatten array
        # at the letter at the end of the array
        array = np.append(arr=array, values=letter)
        samples.append(array)

    os.chdir("..")
    return samples


def get_weights(layers=[2500, 250, 10]):
    '''
    Read the files W1.txt and W2.txt at tmp/ to get the weights
    '''

    W1 = np.zeros((layers[0], layers[1]))
    W2 = np.zeros((layers[1], layers[2]))

    currentDirectory = os.getcwd()  # get the current directory path
    path = os.path.join(currentDirectory, "tmp")
    if os.path.exists(path) == False:  # if folder tmp does not exist, return None
        return None, None

    filePath = os.path.join(path, "W1.txt")
    if os.path.isfile(filePath) == False:
        return None, None  # if file does not exist, return None

    f = open(filePath, "r")

    for i in range(layers[0]):
        for j in range(layers[1]):
            line = f.readline()
            W1[i][j] = float(line)

    f.close()

    filePath = os.path.join(path, "W2.txt")
    if os.path.isfile(filePath) == False:
        return None, None  # if file does not exist, return None

    f = open(filePath, "r")

    for i in range(layers[1]):
        for j in range(layers[2]):
            line = f.readline()
            W2[i][j] = float(line)

    f.close()

    return W1, W2


def save_weights(W1, W2, layers=[2500, 250, 10]):
    '''
    Save the weights in two files named W1.txt and W2.txt at directory tmp/
    '''
    currentDirectory = os.getcwd()
    path = os.path.join(currentDirectory, "tmp")
    if os.path.exists(path) == False:
        os.mkdir(path)  # creates tmp folder if does not exist

    filePath = os.path.join(path, "W1.txt")
    if os.path.isfile(filePath):
        os.remove(filePath)  # deletes the file

    f = open(filePath, "w")

    for i in range(layers[0]):
        for j in range(layers[1]):
            f.write(str(W1[i][j]) + "\n")

    f.close()

    filePath = os.path.join(path, "W2.txt")
    if os.path.isfile(filePath):
        os.remove(filePath)  # deletes the file

    f = open(filePath, "w")

    for i in range(layers[1]):
        for j in range(layers[2]):
            f.write(str(W2[i][j]) + "\n")

    f.close()


def save_numIterations(num):
    '''
    Save the number of iterations in a file named numIterations.txt at directory tmp
    '''
    currentDirectory = os.getcwd()
    path = os.path.join(currentDirectory, "tmp")
    if os.path.exists(path) == False:
        os.mkdir(path)  # creates tmp folder if does not exist

    filePath = os.path.join(path, "numIterations.txt")
    if os.path.isfile(filePath):
        os.remove(filePath)  # deletes the file

    f = open(filePath, "w")

    f.write(str(num) + "\n")

    f.close()


def getIterations():
    '''
    Read the files numIterations.txt at tmp/ to get the number of iterations
    '''
    currentDirectory = os.getcwd()  # get the current directory path
    path = os.path.join(currentDirectory, "tmp")
    if os.path.exists(path) == False:  # if folder tmp does not exist, return None
        return 0

    filePath = os.path.join(path, "numIterations.txt")
    if os.path.isfile(filePath) == False:  # if file does not exist, return None
        return 0

    f = open(filePath, "r")

    line = f.readline()

    f.close()

    return int(line)
