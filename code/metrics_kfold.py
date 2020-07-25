# Libraries
from sklearn.model_selection import KFold
from r_connect               import r_connect
import numpy
import data as DATASETS

'''
    Main code - to be executed
'''
if __name__ == '__main__':
    # Data
    inputs, target = DATASETS.get_dataset('iris')
    
    # K-fold Parameters
    k = 10
    kf = KFold(n_splits=k)
    splits = kf.split(inputs)

    # Store Output
    pos = 0
    measures = [None] * k

    # Connect to R
    connector = r_connect()
    
    for train_index, test_index in splits:
        X_train, X_test = inputs[train_index], inputs[test_index]
        y_train, y_test = target[train_index], target[test_index]

        # Compute and print metrics for dataset
        measures[pos] = connector.get_metrics(X_train, y_train)
        pos += 1

    # Measure from the whole dataset
    final_measure = connector.get_metrics(inputs, target)

    # TODO: Look at
    # https://scikit-learn.org/stable/auto_examples/datasets/plot_iris_dataset.html
 