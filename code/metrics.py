# Libraries
from r_connect import r_connect
from numpy     import matrix
import data as DATASETS
import numpy

'''
    Main code - to be executed
'''
if __name__ == '__main__':
    #Data
    inputs, target = DATASETS.get_dataset('iris')
    
    # Connect to R
    connector = r_connect()
    # Compute and print metrics for dataset
    connector.get_print_metrics(inputs, target)

    # TODO: Look at
    # https://scikit-learn.org/stable/auto_examples/datasets/plot_iris_dataset.html
 