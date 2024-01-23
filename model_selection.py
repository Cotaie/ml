import random
from neural.constants import SPLIT_IDX

X = [[1,2],[3,4],[5,6],[7,8],[9,10]]
Y = [[1], [2], [3], [4], [5]]

nr_examples = len(X)
indices = list(range(nr_examples))

#split_idx = 

print(indices)
random.shuffle(indices)
print(indices)

def split_train_validation(X, Y):
    nr_examples = len(X)
    indices = list(range(nr_examples))
    random.shuffle(indices)
    validation_idx = int(SPLIT_IDX * nr_examples)
    return {
        "training": ([X[i] for i in indices[:validation_idx]], [Y[i] for i in indices[:validation_idx]]),
        "test": ([X[i] for i in indices[validation_idx:]], [Y[i] for i in indices[validation_idx:]])
    }

data = split_train_validation(X, Y)

print("training: ", data['training'])
print("test: ", data['test'])