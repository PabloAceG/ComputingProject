# Libraries
from sklearn.model_selection import KFold
from sklearn.naive_bayes     import GaussianNB
from sklearn.tree            import DecisionTreeClassifier
from sklearn.neighbors       import NearestCentroid

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
        kf = KFold(n_splits=k)
        splits = kf.split(inputs)
        #                   Precision Recall Fallout Balanced F1 MCC AUC
        mean_naive = np.array([0,       0,      0,      0,    0,  0,  0])
        mean_tree  = np.array([0,       0,      0,      0,    0,  0,  0])
        mean_knn   = np.array([0,       0,      0,      0,    0,  0,  0])
        
        # Get measurements using K-fold
        for train_index, test_index in splits:
            # Data partition
            x_train, x_test = inputs[train_index], inputs[test_index]
            y_train, y_test = target[train_index], target[test_index]

            filename = d + '-k' + str(k)

            print(((y_train == 0).sum(), (y_train == 1).sum()))

            # Train NN
            print('---> Naive Bayes')
            clf   = GaussianNB() 
            predictions   = DATASETS.train_predict(clf, x_train, y_train, x_test)
            naive_metrics = DATASETS.calculate_results(y_test, predictions)
            mean_naive = np.add(mean_naive, naive_metrics)
            naive_metrics = ['Naive Bayes'] + naive_metrics
            DATASETS.store_results(filename, naive_metrics)

            print('---> Decision Tree')
            clf   = DecisionTreeClassifier()
            predictions  = DATASETS.train_predict(clf, x_train, y_train, x_test)
            tree_metrics = DATASETS.calculate_results(y_test, predictions)
            mean_tree = np.add(mean_tree, tree_metrics)
            tree_metrics = ['Decision Tree'] + tree_metrics
            DATASETS.store_results(filename, tree_metrics)
            
            print('---> Nearest Centroid')
            clf   = NearestCentroid()
            predictions = DATASETS.train_predict(clf, x_train, y_train, x_test)
            knn_metrics = DATASETS.calculate_results(y_test, predictions)
            mean_knn = np.add(mean_knn, knn_metrics)
            knn_metrics = ['Nearest Centroid'] + knn_metrics
            DATASETS.store_results(filename, knn_metrics)

            print('------------------------------')

        # K-fold Mean
        mean_naive = [round(i, 4) for i in (mean_naive / k)]
        DATASETS.store_results(filename, ['Naive Bayes Mean'] + mean_naive)
        mean_tree = [round(i, 4) for i in (mean_tree / k)]
        DATASETS.store_results(filename, ['Decision Tree Mean'] + mean_tree)
        mean_knn = [round(i, 4) for i in (mean_knn / k)]
        DATASETS.store_results(filename, ['Nearest Centroid Mean'] + mean_knn)

        print([mean_naive, mean_tree, mean_knn])
    
