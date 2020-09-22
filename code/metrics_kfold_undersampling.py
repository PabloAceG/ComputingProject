# Libraries
from sklearn.model_selection import KFold
from sklearn.naive_bayes     import GaussianNB
from sklearn.tree            import DecisionTreeClassifier
from sklearn.neighbors       import NearestCentroid

from imblearn.datasets       import make_imbalance
from imblearn.under_sampling import NearMiss
from imblearn.pipeline       import make_pipeline

from r_connect import r_connect
import data as DATASETS

if __name__ == '__main__':
    # Data
    dataset = 'apache'
    inputs, target = DATASETS.get_dataset(dataset)
    
    # K-fold Parameters
    k = 5
    kf = KFold(n_splits=k, shuffle=True, random_state=1)
    splits = kf.split(inputs)

    RANDOM_STATE = 42
    
    # Get measurements using K-fold
    for train_index, test_index in splits:
        # Data partition
        x_train, x_test = inputs[train_index], inputs[test_index]
        y_train, y_test = target[train_index], target[test_index]

        X, Y = make_imbalance(
            x_train, y_train,
            sampling_strategy={0:50, 1:50},
            random_state=RANDOM_STATE)

        filename = dataset + '-k' + str(k) + '-under'

        # Train NN
        print('---> Naive Bayes')
        clf   = GaussianNB() 
        pipeline = make_pipeline(
            NearMiss(version=2),
            clf)
        predictions   = DATASETS.train_predict(pipeline, X, Y, x_test)
        naive_metrics = ['Naive Bayes'] + DATASETS.calculate_results(y_test, predictions)
        DATASETS.store_results(filename, naive_metrics)

        print('---> Decision Tree')
        clf   = DecisionTreeClassifier()
        pipeline = make_pipeline(
            NearMiss(version=2),
            clf)
        predictions  = DATASETS.train_predict(pipeline, X, Y, x_test)
        tree_metrics = ['Decision Tree'] + DATASETS.calculate_results(y_test, predictions)
        DATASETS.store_results(filename, tree_metrics)
        
        print('---> Nearest Centroid')
        clf   = NearestCentroid()
        pipeline = make_pipeline(
            NearMiss(version=2),
            clf)
        predictions = DATASETS.train_predict(pipeline, X, Y, x_test)
        knn_metrics = ['Nearest Centroid'] + DATASETS.calculate_results(y_test, predictions)
        DATASETS.store_results(filename, knn_metrics)

        print('------------------------------')
