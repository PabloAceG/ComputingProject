# Libraries
from r_connect import r_connect
from numpy     import matrix
import numpy
import csv

# Global variables
DATASET_PATH = './dataset/iris.csv'
DATASET = []

'''
    Load Dataset data into an array 
'''
with open (DATASET_PATH, 'r') as csv_file:
    # Skip header
    next (csv_file)
    # Interator object to read the CSV
    csv_reader = csv.reader (
        csv_file, 
        delimiter=',', 
        quoting=csv.QUOTE_ALL
    )
    # Create array from CSV
    for row in csv_reader :
        DATASET.append (row)

def parse_dataset () : 
    ## Data
    # Input 
    X = numpy.array (DATASET) # Transformed to numpy array to allow more 
    X = X[:, 0 : -1]          # operations on it.
    X = numpy.array ( [ 
        numpy.array (row).astype (numpy.float) 
        for row in X 
    ] )
    # Target
    targets = {'setosa': 1, 'versicolor': 2, 'virginica': 3} 
    Y = numpy.array( [
        targets[row[-1]] for row in DATASET
    ] )

    return X, Y

'''
    Main code - to be executed
'''
if __name__ == '__main__':
    #Data
    inputs, target = parse_dataset ()
    
    # Connect to R
    connector = r_connect()
    # Compute and print metrics for dataset
    connector.get_print_metrics(inputs, target)

    # TODO: Look at
    # https://scikit-learn.org/stable/auto_examples/datasets/plot_iris_dataset.html
 