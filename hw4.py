import numpy as np


# Function for preparing data
def prepare_data(msg_list):
    types = np.array([msg.split('\t')[0] for msg in msg_list])
    messages = np.array([msg.split('\t')[1:] for msg in msg_list])
    return types, np.hstack(messages)


# Function to get bag of words
def vectorize(msg_list):
    messages = np.array([msg.split() for msg in msg_list])
    dictionary = np.unique(np.hstack(np.array(messages)))
    result_matrix = np.array([[message.count(word) for word in dictionary] for message in messages])
    return result_matrix


class NaiveBayes:
    def __init__(self, alpha):
        self.alpha = alpha
        self.types = np.zeros(0)
        self.p = np.zeros(0)
        self.freq = np.zeros(0)
        self.words_amount = np.zeros(0)
        self.dict_size = 0

    def fit(self, x, y):
        self.dict_size = x.shape[1]
        self.types, self.p = np.unique(y, return_counts=True)
        self.p = self.p / y.shape[0]
        types_cnt = self.types.shape[0]

        self.freq = np.zeros(shape=(types_cnt, self.dict_size))
        self.words_amount = np.zeros(types_cnt)

        for i in range(types_cnt):
            x_new = x[y == self.types[i]]
            self.words_amount[i] = np.sum(x_new)
            for j in range(self.dict_size):
                self.freq[i, j] = (np.sum(x_new[:, j]) + self.alpha) / \
                                  (self.words_amount[i] + self.alpha * self.dict_size)


    def predict_for_one(self, x):
        print(x)
        n = np.sum(x)
        x_new = x[x > 0]
        sentence = np.log(self.p) + n * np.log(n) - np.sum(x_new * np.log(x_new))  + \
                   np.sum(np.log(self.freq[:, x > 0]) * x_new, axis=1)
        return self.types[np.argmax(sentence)]


    def predict(self, x):
        return np.array([self.predict_for_one(el) for el in x])


    def score(self, X, y):
        result = self.predict(X)
        eq_count = np.sum(result == y)
        return eq_count / y.shape[0]


def main():
    with open('dataset/spam', 'r', encoding='utf-8') as f:
        msg_list = f.read().split('\n')
        msg_cnt = len(msg_list)
        train_size = round(msg_cnt * 0.75)
        types, messages = prepare_data(msg_list)
        result_matrix = vectorize(messages)
        train = result_matrix[:train_size]
        test = result_matrix[train_size:]

        bayes = NaiveBayes(0.000001)
        bayes.fit(train, types[:train_size])
        print(bayes.score(test, types[train_size:]))


if __name__ == "__main__":
    main()