# Libraries
from r_connect import r_connect
from numpy     import matrix
import numpy
import csv

# Global variables
dataset_path = './dataset/iris.csv'
dataset = []

'''
    Load Dataset data into an array 
'''
with open (dataset_path, 'r') as csv_file:
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
        dataset.append (row)

def parse_dataset () : 
    ## Data
    # Input 
    X = numpy.array (dataset) # Transformed to numpy array to allow more 
    X = X[:, 0 : -1]          # operations on it.
    X = numpy.array ( [ 
        numpy.array (row).astype (numpy.float) 
        for row in X 
    ] )
    # Target
    Y = numpy.array( [
        row[-1] for row in dataset
    ] )

    return X, Y

'''
    Main code - to be executed
'''
if __name__ == '__main__':
    #Data
    inputs, target = parse_dataset ()

    # R does not take string values. So each class is translated into a 
    # numerical value.
    for row in range (len(target)) :
        if target[row] == 'setosa' :
            target[row] = 1
        elif target[row] == 'versicolor' :
            target[row] = 2
        elif target[row] == 'virginica' :
            target[row] = 3
        else :
            target[row] = 0
    
    # Connect to R
    connector = r_connect()
    # Compute and print metrics for dataset
    connector.get_print_metrics(inputs, target)

    # TODO: Look at
    # https://scikit-learn.org/stable/auto_examples/datasets/plot_iris_dataset.html
 