import numpy as np
from matplotlib import pyplot as plt


# Function that read data from csv file to numpy matrix and shuffle objects
def read_data(data_file):
    data_matrix = np.loadtxt(data_file, delimiter=',')
    np.random.shuffle(data_matrix)

    data = normalize(data_matrix)

    x = data_matrix[:, :-1]
    y = data_matrix[:, -1]
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
    train_size = int(x.shape[0] * ratio)

    x_train = x[:train_size, :]
    x_test = x[train_size:, :]

    y_train = y[:train_size]
    y_test = y[train_size:]

    return x_train, y_train, x_test, y_test


class NormalLR:
    def __init__(self, tau):
        self.weights = np.zeros(0)
        self.tau = tau

    def fit(self, X, y):
        weights = ((np.linalg.inv(X.T.dot(X) + self.tau * np.eye(X.shape[1]))).dot(X.T)).dot(y)
        self.weights = weights
        return self

    def predict(self, X):
        return X.dot(self.weights)


# Distance function for gradientDescent
def euclidean_dist(x, y):
    return np.sum((x - y)**2)**(1/2)


class GradientLR(NormalLR):
    def __init__(self, *, alpha, tau):
        if alpha <= 0:
            raise ValueError("alpha should be positive")
        self.alpha = alpha
        self.threshold = alpha / 100
        self.tau = tau

    def fit(self, X, y):
        attributes_cnt = X.shape[1]

        weights = (1 / attributes_cnt) * np.random.random_sample(attributes_cnt) - 1 / (2 * attributes_cnt)

        # Calculate weights using loss function until they become stable
        while (True):
            new_weights = weights - self.alpha / X.shape[0] * np.dot((X.dot(weights.T) - y), X)  - self.tau * weights

            if euclidean_dist(weights, new_weights) < self.threshold:
                break
            weights = new_weights

        # Fill self.weights variable
        self.weights = weights

        return self


def sample(size, *, weights):
    X = np.ones((size, 2))
    X[:, 1] = np.random.gamma(4., 2., size)
    y = X.dot(np.asarray(weights))
    y += np.random.normal(0, 1, size)
    return X[:, 1:], y


def mse(y_pred, y_true):
    size = y_pred.shape[0]
    error_array = np.zeros(size)
    for i in range(size):
        error_array[i] = (y_pred[i] - y_true[i]) ** 2
    return np.average(error_array)


def main():
    # Read data from file to numpy matrix and numpy array
    x_data, y_data = read_data("dataset/boston_prices.csv")

    # Split data to train and test
    x_data_train, y_data_train, x_data_test, y_data_test = train_test_split(x_data, y_data, 0.75)

    lr = NormalLR(0)
    lr.fit(x_data_train, y_data_train)
    y_predict = lr.predict(x_data_test)

    grad_lr = GradientLR(alpha=0.001, tau=10**(-6))
    grad_lr.fit(x_data_train, y_data_train)
    y_predict2 = grad_lr.predict(x_data_test)

    print("NormalLR mse on test:")
    print(mse(y_predict, y_data_test))
    print("GradientLR mse on test:")
    print(mse(y_predict2, y_data_test))

    #Second part
    size_list = [128, 256, 512, 1024]
    for el in size_list:
        X, y_true = sample(el, weights=[24., 42.])
        lr.fit(X, y_true)

        plt.scatter(X, y_true)
        plt.plot(X, lr.predict(X), color="red")
        plt.show()

        grad_lr.fit(X, y_true)
        plt.scatter(X, y_true)
        plt.plot(X, grad_lr.predict(X), color="red")
        plt.show()

        print("For size ", el)
        print("NormalLR mse: ", mse(lr.predict(X), y_true))
        print("GradientlLR mse: ", mse(grad_lr.predict(X), y_true))


if __name__ == "__main__":
    main()