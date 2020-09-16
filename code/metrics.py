# Libraries
from r_connect import r_connect
from numpy     import matrix
import data as DATASETS
import numpy

if __name__ == '__main__':
    '''
        Main code - to be executed
    '''
    #Data
    inputs, target = DATASETS.get_dataset('iris')
    
    # Connect to R
    connector = r_connect()
    # Compute and print metrics for dataset
    metrics = connector.get_metrics(inputs, target)

    print(metrics)

    # TODO: Look at
    # https://scikit-learn.org/stable/auto_examples/datasets/plot_iris_dataset.html
 