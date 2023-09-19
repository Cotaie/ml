import random

def split_train_validation(X, Y, split_idx=0.8):
    nr_examples = len(X)
    indices = list(range(nr_examples))
    random.shuffle(indices)
    validation_idx = int(split_idx * nr_examples)
    return {
        "training": ([X[i] for i in indices[:validation_idx]], [Y[i] for i in indices[:validation_idx]]),
        "test": ([X[i] for i in indices[validation_idx:]], [Y[i] for i in indices[validation_idx:]])
    }