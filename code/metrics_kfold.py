# Libraries
from sklearn.model_selection import KFold
from sklearn.naive_bayes     import GaussianNB
from sklearn.tree            import DecisionTreeClassifier
from sklearn.neighbors       import NearestCentroid

from r_connect               import r_connect
import data as DATASETS

if __name__ == '__main__':
    # Data
    dataset = 'apache'
    inputs, target = DATASETS.get_dataset(dataset)
    
    # K-fold Parameters
    k = 5
    kf = KFold(n_splits=k)
    splits = kf.split(inputs)
    
    # Get measurements using K-fold
    for train_index, test_index in splits:
        # Data partition
        x_train, x_test = inputs[train_index], inputs[test_index]
        y_train, y_test = target[train_index], target[test_index]

        filename = dataset + '-k' + str(k)

        # Train NN
        print('---> Naive Bayes')
        clf   = GaussianNB() 
        predictions   = DATASETS.train_predict(clf, x_train, y_train, x_test)
        naive_metrics = ['Naive Bayes'] + DATASETS.calculate_results(y_test, predictions)
        DATASETS.store_results(filename, naive_metrics)

        print('---> Decision Tree')
        clf   = DecisionTreeClassifier()
        predictions  = DATASETS.train_predict(clf, x_train, y_train, x_test)
        tree_metrics = ['Decision Tree'] + DATASETS.calculate_results(y_test, predictions)
        DATASETS.store_results(filename, tree_metrics)
        
        print('---> Nearest Centroid')
        clf   = NearestCentroid()
        predictions = DATASETS.train_predict(clf, x_train, y_train, x_test)
        knn_metrics = ['Nearest Centroid'] + DATASETS.calculate_results(y_test, predictions)
        DATASETS.store_results(filename, knn_metrics)

        print('------------------------------')
    
