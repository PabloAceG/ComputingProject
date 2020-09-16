# Libraries
from sklearn.model_selection import KFold
from sklearn.naive_bayes     import GaussianNB
from r_connect               import r_connect
import data as DATASETS
import numpy
import math

if __name__ == '__main__':
    # Data
    # ant dataset resolves with 0 for the false negatives
    inputs, target = DATASETS.get_dataset('apache', True)
    
    # K-fold Parameters
    k = 5
    kf = KFold(n_splits=k)
    splits = kf.split(inputs)

    # Store Output
    pos = 0
    measures = [None] * k

    # Connect to R
    connector = r_connect()
    
    # Get measurements using K-fold
    for train_index, test_index in splits:
        # Data partition
        x_train, x_test = inputs[train_index], inputs[test_index]
        y_train, y_test = target[train_index], target[test_index]

        # Train NN
        clf = GaussianNB()
        clf.fit(x_train, y_train)
        # Test NN and get confusion matrix
        (true_positive, true_negative, false_positive, false_negative) = DATASETS.confusion_matrix(x_test, y_test, clf)
        print((true_positive, true_negative, false_positive, false_negative))
        print("precision: " + str(DATASETS.precision(true_positive, false_positive)))
        print("recall: "    + str(DATASETS.recall(true_positive, false_negative)))
        print("f1: "        + str(DATASETS.f1_score(true_positive, false_positive, false_negative)))
        print("mcc: "       + str(DATASETS.mcc(true_positive, true_negative, false_positive, false_negative)))

        # Compute and print metrics for dataset
        #measures[pos] = connector.get_metrics(x_train, y_train)
        #pos += 1

    # Measure from the whole dataset
    #final_measure = connector.get_metrics(inputs, target)
