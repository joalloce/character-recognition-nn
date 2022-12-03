import numpy as np
import fileManager

letters = ["B", "C", "D", "F", "G", "H", "J", "K", "L", "M"]


def split_data_and_labels(samples):
    '''
    Split the samples in labels and data.
    The label is the last element in each array.
    '''
    data = samples[:, :-1]
    labels = samples[:, -1]

    return labels, data


def print_sample(sample):
    '''
    Print a sample.
    '''
    row = ""
    for i in range(len(sample)):
        if sample[i] == 1.0:
            row += "*"
        else:
            row += " "
        if (i+1) % 50 == 0:
            print(row)
            row = ""


def reshape_output(Y):
    '''
    Reshape the output in an array.
    '''
    categories = []
    length = len(letters)
    for i in range(len(Y)):
        array = np.zeros(length)  # creates an arrays of zeros
        # 1 in the position where the letter is
        array[letters.index(Y[i])] = 1.0
        categories.append(array)

    categories = np.array(categories)

    return categories


def get_letter_from_index(index):
    '''
    Return the letter from the index.
    '''
    if index >= 0 and index < len(letters):
        return letters[index]

    return "-"  # returns '-' if not valid


def load():
    '''
    Load the samples.
    '''
    test_samples = fileManager.get_samples("./test_samples")
    train_samples = fileManager.get_samples("./train_samples")

    test_labels, test_data = split_data_and_labels(test_samples)
    train_labels, train_data = split_data_and_labels(train_samples)

    test_data = test_data.astype(np.float64)
    train_data = train_data.astype(np.float64)
    train_X = train_data
    test_X = test_data

    train_Y = reshape_output(train_labels)
    test_Y = reshape_output(test_labels)

    return train_X, train_Y, test_X, test_Y
