# Libraries
from sklearn.model_selection import KFold
from sklearn.naive_bayes     import GaussianNB
from sklearn.tree            import DecisionTreeClassifier
from sklearn.neighbors       import NearestCentroid

from imblearn.datasets       import make_imbalance
from imblearn.under_sampling import NearMiss
from imblearn.pipeline       import make_pipeline

import numpy as np

import data as DATASETS

if __name__ == '__main__':
    # Data
    datasets = [
        'ant',
        'apache',
        'camel',
        'ivy',
        'jedit',
        'log4j',
        'poi',
        'synapse',
        'xalan',
        'xerces',
        'hadoop-1',
        'hadoop-2',
        'hadoop-3',
        'hadoop-4',
        'hadoop-5',
        'hadoop-6',
        'hadoop-7',
        'hadoop-8'
    ]

    for d in datasets:
        print('----------' + d + '----------')
        inputs, target = DATASETS.get_dataset(d)
    
        # K-fold Parameters
        k = 5
        kf = KFold(n_splits=k, shuffle=True, random_state=1)
        splits = kf.split(inputs)
        #                   Precision Recall Fallout Balanced F1 MCC AUC
        mean_naive = np.array([0,       0,      0,      0,    0,  0,  0])
        mean_tree  = np.array([0,       0,      0,      0,    0,  0,  0])
        mean_knn   = np.array([0,       0,      0,      0,    0,  0,  0])

        RANDOM_STATE = 42
        
        # Get measurements using K-fold
        for train_index, test_index in splits:
            # Data partition
            x_train, x_test = inputs[train_index], inputs[test_index]
            y_train, y_test = target[train_index], target[test_index]

            X, Y = make_imbalance(
                x_train, y_train,
                sampling_strategy={0:10, 1:10},
                #sampling_strategy={0:50, 1:50},
                random_state=RANDOM_STATE)

            filename = d + '-k' + str(k) + '-under'

            # Train NN
            print('---> Naive Bayes')
            clf   = GaussianNB() 
            pipeline = make_pipeline(
                NearMiss(version=2),
                clf)
            predictions   = DATASETS.train_predict(pipeline, X, Y, x_test)
            naive_metrics = DATASETS.calculate_results(y_test, predictions)
            mean_naive = np.add(mean_naive, naive_metrics)
            naive_metrics = ['Naive Bayes'] + naive_metrics
            DATASETS.store_results(filename, naive_metrics)

            print('---> Decision Tree')
            clf   = DecisionTreeClassifier()
            pipeline = make_pipeline(
                NearMiss(version=2),
                clf)
            predictions  = DATASETS.train_predict(pipeline, X, Y, x_test)
            tree_metrics = DATASETS.calculate_results(y_test, predictions)
            mean_tree = np.add(mean_tree, tree_metrics)
            tree_metrics = ['Decision Tree'] + tree_metrics
            DATASETS.store_results(filename, tree_metrics)
            
            print('---> Nearest Centroid')
            clf   = NearestCentroid()
            pipeline = make_pipeline(
                NearMiss(version=2),
                clf)
            predictions = DATASETS.train_predict(pipeline, X, Y, x_test)
            knn_metrics = DATASETS.calculate_results(y_test, predictions)
            mean_knn = np.add(mean_knn, knn_metrics)
            knn_metrics = ['Nearest Centroid'] + knn_metrics
            DATASETS.store_results(filename, knn_metrics)

            print('------------------------------')

        # K-fold Mean
        mean_naive = mean_naive / k
        DATASETS.store_results(filename, ['Naive Bayes Mean'] + mean_naive.tolist())
        mean_tree = mean_tree / k
        DATASETS.store_results(filename, ['Decision Tree Mean'] + mean_tree.tolist())
        mean_knn = mean_knn / k
        DATASETS.store_results(filename, ['Nearest Centroid Mean'] + mean_knn.tolist())

        print([mean_naive, mean_tree, mean_knn])
