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
    X = numpy.array (dataset) # Transformed to numpy array to allow more operations on it.
    X = X[:, 0 : -1]
    X = [ 
        numpy.array (row).astype (numpy.float) 
        for row in X 
    ]
    # Target
    Y = [
        row[-1] for row in dataset
    ]

    return X, Y

'''
    Main code - to be executed
'''
if __name__ == '__main__':
    #Data
    inputs, target = parse_dataset ()

    print (len(inputs))
    print (len(inputs[0]))
    print (len(target))


    # Connect to R
    connector = r_connect()
    # Get metrics for dataset
    connector.get_metrics(inputs, target)

    # TODO: Look at
    # https://scikit-learn.org/stable/auto_examples/datasets/plot_iris_dataset.html
 