# Libraries
from r_connect import r_connect
from numpy     import matrix
import numpy
import data as DATASETS

if __name__ == '__main__':
    '''
        Main code - to be executed
    '''

    #Data
    inputs, target = DATASETS.get_dataset('ant')
    # Connect to R
    connector = r_connect()
    # Compute and print metrics for dataset
    metrics_iris = connector.get_metrics(inputs, target)

    #Data
    inputs, target = DATASETS.get_dataset('apache')
    # Connect to R
    connector = r_connect()
    # Compute and print metrics for dataset
    metrics_apache = connector.get_metrics(inputs, target)

    #Data
    inputs, target = DATASETS.get_dataset('camel')
    # Connect to R
    connector = r_connect()
    # Compute and print metrics for dataset
    metrics_camel = connector.get_metrics(inputs, target)

    #Data
    inputs, target = DATASETS.get_dataset('ivy')
    # Connect to R
    connector = r_connect()
    # Compute and print metrics for dataset
    metrics_ivy = connector.get_metrics(inputs, target)

    #Data
    inputs, target = DATASETS.get_dataset('jedit')
    # Connect to R
    connector = r_connect()
    # Compute and print metrics for dataset
    metrics_jedit = connector.get_metrics(inputs, target)

    #Data
    inputs, target = DATASETS.get_dataset('log4j')
    # Connect to R
    connector = r_connect()
    # Compute and print metrics for dataset
    metrics_log4j = connector.get_metrics(inputs, target)

    #Data
    inputs, target = DATASETS.get_dataset('synapse')
    # Connect to R
    connector = r_connect()
    # Compute and print metrics for dataset
    metrics_synapse = connector.get_metrics(inputs, target)

    #Data
    inputs, target = DATASETS.get_dataset('xalan')
    # Connect to R
    connector = r_connect()
    # Compute and print metrics for dataset
    metrics_xalan = connector.get_metrics(inputs, target)

    #Data
    inputs, target = DATASETS.get_dataset('xerces')
    # Connect to R
    connector = r_connect()
    # Compute and print metrics for dataset
    metrics_xerces = connector.get_metrics(inputs, target)

    # TODO: Look at
    # https://scikit-learn.org/stable/auto_examples/datasets/plot_iris_dataset.html
 