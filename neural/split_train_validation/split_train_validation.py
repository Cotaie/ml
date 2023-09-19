import random

def split_train_validation(X, Y, split_idx=0.8, seed=42):
    nr_examples = len(X)
    indices = list(range(nr_examples))
    random.seed(seed)
    random.shuffle(indices)
    validation_idx = int(split_idx * nr_examples)
    return {
        "training": ([X[i] for i in indices[:validation_idx]], [Y[i] for i in indices[:validation_idx]]),
        "validation": ([X[i] for i in indices[validation_idx:]], [Y[i] for i in indices[validation_idx:]])
    }