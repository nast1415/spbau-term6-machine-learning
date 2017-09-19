import numpy as np
import pandas as pd
from PIL import Image, ImageDraw


# Function for reading data from csv file
def read_data(path):
    monsters_df = pd.read_csv(path)
    return monsters_df[:370], monsters_df[370:]


# Function for calculating entropy
def entropy(data):
    _, counts = np.unique(data, return_counts=True)
    probabilities = counts / len(data)
    return -probabilities.dot(np.log(probabilities))


# Function for separating data by predicate
def separate(data, predicate):
    pred_attr = predicate[0]
    pred_value = predicate[1]
    pred_type = predicate[2]

    if pred_type == 0:
        data_true = data[data[pred_attr] <= pred_value]
        data_false = data[data[pred_attr] > pred_value]
    else:
        data_true = data[data[pred_attr] == pred_value]
        data_false = data[data[pred_attr] != pred_value]
    return data_true, data_false


# Score function
def score(data, predicate):
    attr = predicate[0]
    value = predicate[1]

    data_true = data[data[attr] <= value]
    data_false = data[data[attr] > value]

    total_count = np.array(data[attr]).size
    true_count = np.array(data_true[attr]).size
    false_count = np.array(data_false[attr]).size

    return entropy(data['type']) - float(true_count * entropy(data_true['type']) +
                                         false_count * entropy(data_false['type'])) / total_count


# Function to get attribute type (0 for float and 1 for string)
def attribute_type(attr):
    attr_type = 0
    try:
        x = float(attr)
        return x, attr_type
    except ValueError:
        attr_type = 1
        return attr, attr_type


# Main decision tree class
class DecisionTree:
    # Build function for decision tree (ID3)
    def build(self, x, score_func):
        y = np.array(x['type'])
        # Check if all data are in the same class. If it is true, return leaf with string name of this class
        if np.unique(y).shape[0] == 1:
            return Leaf(y[0])

        # If it is not true, find most informative predicate
        attributes = x.dtypes.index[:-1]

        max_info_gain = 0

        for attr in attributes:
            attr_values = np.unique(x[attr])
            attr_val, attr_type = attribute_type(attr_values[0])

            for value in attr_values:
                info_gain = score_func(x, [attr, value, attr_type])
                if info_gain > max_info_gain:
                    max_info_gain = info_gain
                    most_inf_predicate = [attr, value, attr_type]

        # Separate data by most informative predicate
        left_subtree, right_subtree = separate(x, most_inf_predicate)


        # Check if one of the subtrees is empty
        if left_subtree.empty or right_subtree.empty:
            # If one of the subtrees is empty return leaf with the string name of the attribute = Majority(x)
            class_names, counts = np.unique(y, return_counts=True)
            max_number = np.max(counts)
            for i in range(class_names.shape[0]):
                if counts[i] == max_number:
                    return Leaf(class_names[i])
        else:
            # If both subtrees are not empty, then return inner Node with predicate and recursive subtrees
            return Node(
                most_inf_predicate[0], most_inf_predicate[1], most_inf_predicate[2],
                DecisionTree().build(right_subtree, score_func),
                DecisionTree().build(left_subtree, score_func))

    # Predict function for decision tree
    def predict(self, x):
        if isinstance(self, Node):
            pred_attr = self.predicate_attr
            pred_val = self.predicate_val
            pred_type = self.predicate_type

            attr_val = np.array(x[pred_attr])
            if pred_type == 0:
                if attr_val[0] <= pred_val:
                    return self.true_branch.predict(x)
                else:
                    return self.false_branch.predict(x)
            else:
                if attr_val[0] == pred_val:
                    return self.true_branch.predict(x)
                else:
                    return self.false_branch.predict(x)

        elif isinstance(self, Leaf):
            return self.class_name
        return 1


# Class for decision tree node
class Node(DecisionTree):
    def __init__(self, predicate_1, predicate_2, predicate_3, false_branch, true_branch):
        self.predicate_attr = predicate_1
        self.predicate_val = predicate_2
        self.predicate_type = predicate_3
        self.false_branch = false_branch
        self.true_branch = true_branch


class Leaf(DecisionTree):
    def __init__(self, class_name):
        self.class_name = class_name


def getdepth(tree):
    if isinstance(tree, Node):
        return 1 + max(getdepth(tree.false_branch), getdepth(tree.true_branch))
    else:
        return 1


def getwidth(tree):
    if isinstance(tree, Node):
        return getwidth(tree.false_branch) + getwidth(tree.true_branch)
    else:
        return 1

def drawtree(tree, path='tree.jpg'):
    w = getwidth(tree) * 100
    h = getdepth(tree) * 100

    img = Image.new('RGB', (w, h), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    drawnode(draw, tree, w / 2, 20)
    img.save(path, 'JPEG')


def drawnode(draw, tree, x, y):
    if isinstance(tree, Node):
        shift = 100
        width1 = getwidth(tree.false_branch) * shift
        width2 = getwidth(tree.true_branch) * shift
        left = x - (width1 + width2) / 2
        right = x + (width1 + width2) / 2

        if tree.predicate_type == 0:
            predicate = str(tree.predicate_attr) + "<=" + str(tree.predicate_val)
        else:
            predicate = str(tree.predicate_attr) + "==" + str(tree.predicate_attr)

        draw.text((x - 20, y - 10), predicate, (0, 0, 0))
        draw.line((x, y, left + width1 / 2, y + shift), fill=(255, 0, 0))
        draw.line((x, y, right - width2 / 2, y + shift), fill=(255, 0, 0))
        drawnode(draw, tree.false_branch, left + width1 / 2, y + shift)
        drawnode(draw, tree.true_branch, right - width2 / 2, y + shift)
    elif isinstance(tree, Leaf):
        draw.text((x - 20, y), tree.class_name, (0, 0, 0))


def main():
    train_x, test = read_data("dataset/monsters.csv")
    print(train_x)
    dt = DecisionTree().build(train_x, score)
    print(dt.predict(test[:1]))
    drawtree(dt)


if __name__ == "__main__":
    main()