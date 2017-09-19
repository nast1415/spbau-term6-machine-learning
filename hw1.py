import numpy as np


# Function that read data from csv file to numpy matrix and shuffle objects
def read_data(data_file):
    wine_matrix = np.loadtxt(data_file, delimiter=',')
    np.random.shuffle(wine_matrix)

    x = wine_matrix[:, 1:]
    y = wine_matrix[:, 0]
    return x, y


# Function to normalize data matrix
def normalize(data):
    max_array = np.max(data, axis=0)
    min_array = np.min(data, axis=0)
    max_array_size = len(max_array)

    for i in range(data.shape[0]):
        for j in range(max_array_size):
            data[i][j] = (data[i][j] - min_array[j]) / max_array[j]
    return data


# Function to split data matrix to train and test
def train_test_split(x, y, ratio):
    x = normalize(x)
    train_size = int(x.shape[0] * ratio)

    x_train = x[:train_size, :]
    x_test = x[train_size:, :]

    y_train = y[:train_size]
    y_test = y[train_size:]

    return x_train, y_train, x_test, y_test


# Two distance functions for knn
def euclidean_dist(x, y):
    return np.sum((x - y)**2)**(1/2)


def manhattan_dist(x, y):
    return np.sum(np.abs(x - y))


# Supporting function that is used in knn function
def find_k_nearest_neighbour(x_train, y_train, test, k, dist):
    distance = np.apply_along_axis(dist, 1, x_train, test)
    k_nearest = distance.argsort()[:k]

    results = np.zeros(k)
    for i in range(k):
        ind = k_nearest[i]
        results[i] = y_train[ind]
    results = results.astype(int)
    counts = np.bincount(results)

    return np.argmax(counts)


# k nearest neighbours classifier
def knn(x_train, y_train, x_test, k, dist):
    y_pred = np.zeros(len(x_test))
    for i in range(len(x_test)):
        y_pred[i] = find_k_nearest_neighbour(x_train, y_train, x_test[i], k, dist)
    return y_pred


def print_precision_recall(y_pred, y_test):
    n_classes = len(np.unique(y_test))
    test_size = len(y_test)

    for c in range(n_classes):
        tp = 0
        fp = 0
        fn = 0
        for i in range(test_size):
            if y_pred[i] == y_test[i] == c + 1:
                tp += 1
            elif (y_pred[i] != y_test[i]) and (y_pred[i] == c + 1):
                fp += 1
            elif (y_pred[i] != y_test[i]) and (y_test[i] == c + 1):
                fn += 1

        if tp == 0:
            precision = 0
            recall = 0
        else:
            precision = tp / (fp + tp)
            recall = tp / (fn + tp)

        print(c + 1, precision, recall)


# Leave one out cross validation
def loocv(x_train, y_train, dist):
    train_size = x_train.shape[0]
    k_quality = np.zeros(train_size)

    for k in range(train_size - 1):
        for i in range(train_size - 1):
            test = x_train[i]
            test_result = y_train[i]

            x_train_i = np.delete(x_train, i, 0)
            y_train_i = np.delete(y_train, i, 0)

            pred = find_k_nearest_neighbour(x_train_i, y_train_i, test, k + 1, dist)
            if pred == test_result:
                k_quality[k + 1] += 1

    return k_quality.argmax()


def main():
    #Read data from file to numpy matrix and numpy array
    x_data, y_data = read_data("dataset/wine.csv")

    #Split data to train and test
    x_data_train, y_data_train, x_data_test, y_data_test = train_test_split(x_data, y_data, 0.75)

    #Find optimal k for two distance metrics
    k_opt_eucl = loocv(x_data_train, y_data_train, euclidean_dist)
    k_opt_manh = loocv(x_data_train, y_data_train, manhattan_dist)

    #Make prediction
    y_pred_eucl = knn(x_data_train, y_data_train, x_data_test, k_opt_eucl, euclidean_dist)
    y_pred_manch = knn(x_data_train, y_data_train, x_data_test, k_opt_manh, manhattan_dist)

    #Precision recall
    print("Precision recall: euclidean distance")
    print_precision_recall(y_pred_eucl, y_data_test)
    print("Precision recall: manhattan distance")
    print_precision_recall(y_pred_manch, y_data_test)

if __name__ == "__main__":
    main()


