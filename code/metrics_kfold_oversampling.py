# Libraries
from sklearn.model_selection import KFold
from sklearn.naive_bayes     import GaussianNB

from imblearn.pipeline      import make_pipeline
from imblearn.over_sampling import SMOTE

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

    RANDOM_STATE = 42

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

        # Pipeline to NN
        clf = make_pipeline(
            SMOTE(),
            GaussianNB())
        # Train NN
        clf.fit(x_train, y_train)
        # Test NN and get confusion matrix
        predictions = clf.predict(x_test)
        print(DATASETS.confusion_matrix(x_test, y_test, clf))
        print("precision: " + str(DATASETS.precision(y_test, predictions)))
        print("recall: "    + str(DATASETS.recall   (y_test, predictions)))
        print("fall_out: "  + str(DATASETS.fall_out (y_test, predictions)))
        print("balanced: "  + str(DATASETS.balanced (y_test, predictions)))
        print("f1: "        + str(DATASETS.f1       (y_test, predictions)))
        print("mcc: "       + str(DATASETS.mcc      (y_test, predictions)))
        print("auc: "       + str(DATASETS.auc      (y_test, predictions)))

        # Compute and print metrics for dataset
        measures[pos] = connector.get_metrics(x_train, y_train)
        pos += 1

    # Measure from the whole dataset
    final_measure = connector.get_metrics(inputs, target)
