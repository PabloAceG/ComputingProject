from numpy import matrix
import arff
import csv
import numpy
import sys
import random
import math
import pandas as pd

def __load_arff(path):
    '''
        Loads the dataset content of an .arff file into the workspace.
        Input:
            - path: Location of the file.
        Output:
            - dataset: Data from the file.
    '''
    
    dataset = []

    with open(path, 'r') as data:
        dataset = arff.load(data)['data']        

    return dataset

def __load_csv(path):
    '''
        Loads the data content of an .csv file into the workspace.
        Input:
            - path: Location of the file.
        Output:
            - dataset: Data from the file.
    '''

    dataset = []

    with open (path, 'r') as csv_file:
        dataset = pd.read_csv(
            csv_file,
            sep=','
        )

    return dataset

def __parte_dataset(dataset, start):
    '''
        Takes a dataset and parses it to only take what is necessary.
        Inputs:
            - dataset: Input raw data as a Pandas Data Framework.
            - start: Where columns start to be useful.
        Output:
            - input: Input arrays of the dataset.
            - target: Target column of the dataset.
    '''

    last_column = -1
    num_rows    = len(dataset)

    # Input
    input_columns = dataset.columns[start : last_column]
    input = dataset[input_columns].head(num_rows).astype(float).to_numpy()

    # Target
    target_column = dataset.columns[last_column]
    target = dataset[target_column].head(num_rows)

    # Parse output
    sample = target[0]
    if not isinstance(sample, str): 
        target = numpy.where(target > 0, 1, 0)
    else: 
        has_faults = target == 'yes'
        target = numpy.where(has_faults , 1, 0)

    return input, target

def __data_preparation(path, start, type='arff', shuffle=False):
    '''
        Loads an .arff/.csv file and parses its content to input/output valid 
        for a Machine Learning application.
        Input:
            - path: Location of the file.
            - start: Where columns start to be useful (string columns are not 
                necessary).
            - type: It can be:
                - arff
                - csv
        Output:
            - input: Input arrays of the dataset.
            - target: Target column of the dataset.
    '''
    
    # Load data
    dataset = []
    if   type == 'arff':
        dataset = __load_arff(path)
    elif type == 'csv':
        dataset = __load_csv(path)
    else:
        raise Exception('Not a valid file type. Try with arff or csv!')
        sys.exit(404)
    dataset = pd.DataFrame(dataset)

    if shuffle:
        dataset = dataset.sample(frac=1)
    
    # Parse data
    data = (dataset, start)
    input, target = __parte_dataset(*data)
    
    return input, target

def __data_preparation_iris_dataset(path):

    # Load data
    dataset = __load_csv(path)
    dataset = pd.DataFrame(dataset)

    # Parse data
    start    = 0
    last     = -1
    num_rows = len(dataset)

    # Input
    input_columns = dataset.columns[start : last]
    input = dataset[input_columns].head(num_rows).astype(float).to_numpy()

    # Target
    target_column = dataset.columns[last]
    target = dataset[target_column].head(num_rows)
    

    # Parse target
    values = {'setosa': 1, 'versicolor': 2, 'virginica': 3}
    target = numpy.array( [
        values[type] for type in target
    ] )

    return input, target

def get_dataset(name, shuffle=False):
    '''
        Retrieves the data of a given dataset, separated into input information
        and target output - .arff or .csv files only.
        Input:
            - name: Dataset name.
                - ant
                - apache
                - camel
                - iris
                - ivy
                - jedit
                - log4j
                - poi
                - synapse
                - xalan
                - xerces
        Output:
            - input: Input arrays of the dataset.
            - target: Target column of the dataset.
    '''

    path  = ''     # Location of file
    start = 0      # From which columns to use
    type  = 'arff' # Type of file to be read. Default .arff

    if name == 'ant':
        # Retrieve data
        path = './dataset/ant-1.7.arff'
        start = 3

    elif name == 'apache':
        # Retrieve data
        path = './dataset/Apache.csv'
        start = 3
        type = 'csv'
        
    elif name == 'camel':
        # Retrieve data
        path = './dataset/camel-1.6.arff'
        start = 3
        
    elif name == 'iris': # Special case. Non-binary output. Needs different 
                         # treatment.
        # Retrieve data
        path = './dataset/iris.csv'
        input, target = __data_preparation_iris_dataset(path, shuffle)

        return input, target

    elif name == 'ivy':
        # Retrieve data
        path = './dataset/ivy-2.0.arff'
        start = 3

    elif name == 'jedit':
        # Retrieve data
        path = './dataset/jedit-4.3.arff'
        start = 3
        
    elif name == 'log4j':
        # Retrieve data
        path = './dataset/log4j-1.2.arff'
        start = 3

    elif name == 'poi':
        # Retrieve data
        path = './dataset/poi-3.0.arff'
        start = 3
        
    elif name == 'synapse':
        # Retrieve data
        path = './dataset/synapse-1.2.arff'
        start = 3
        
    elif name == 'xalan':
        # Retrieve data
        path = './dataset/xalan-2.7.arff'
        start = 3
        
    elif name == 'xerces':
        # Retrieve data
        path = './dataset/xerces-1.4.arff'
        start = 3
        
    else:
        raise Exception('The given dataset name is not valid.') 
        sys.exit(404)
    
    # Return results
    input, target = __data_preparation(path, start, type, shuffle)

    return input, target

def confusion_matrix(x_test: list, y_test: list, classifier):
    '''
        Obtains the confusion matrix for a given testing dataset, with binary
        output.
        Input:
            - x_test: paremeters for testing dataset.
            - y_test: target/desired output for testing dataset.
            - classifier: trained classifier.
        Output:
            (
                true_positive: successfully predicted positives
                true_negative: successfully predicted negatives
                false_positive: unsuccessfully predicted positives
                false_negative: unsuccessfully predicted positives
            )
    '''
    # Confusion Matrix Cells
    true_positive:float  = 0
    true_negative:float  = 0
    false_positive:float = 0
    false_negative:float = 0

    # Calculate number of repetitions of each classification. 
    for (input, target) in zip(x_test, y_test):
        prediction:int = classifier.predict([input])[0]

        if prediction == target:    # Success
            if prediction: true_positive += 1
            else:          true_negative += 1
        else:                       # Wrong
            if prediction: false_positive += 1
            else:          false_negative += 1
    
    # Transform absolute to relative values
    num_samples = len(y_test)
    true_positive  = true_positive  / num_samples
    true_negative  = true_negative  / num_samples
    false_positive = false_positive / num_samples
    false_negative = false_negative / num_samples

    return (true_positive, true_negative, false_positive, false_negative)

def recall(tp: float, fn: float) -> float:
    '''
        Sensitivity, recall, hit rate or True Positive Rate (RPR)
               TP       TP
        TPR = ---- = --------- = 1 - FNR
               P      TP + FN
    '''
    return tp / (tp + fn)

def precision(tp: float, fp: float) -> float:
    '''
        Precision or positive predictive value (PPV).
                 TP
        PPV = --------- = 1 - FDR
               TP + FP
        Inputs:
            - tp (float): True Positives
            - tn (float): True Negatives
            - fp (float): False Positives
            - fn (float): False Negatives
        Returns:
            - MCC (float)
    '''
    return tp / (tp + fp)

def f1_score(tp: float, fp: float, fn: float) -> float:
    '''
        F1 Score. Is the harmonic mean of precision and sensitivity.
                PPV x TPR         2 TP
        F1 = 2 ----------- = ----------------
                PPV + TPR     2 TP + FP + FN
        Inputs:
            - tp (float): True Positives
            - fp (float): False Positives
            - fn (float): False Negatives
        Returns:
            - F1 (float)
    '''
    return (2 * tp) / ((2 * tp) + fp + fn)

def mcc(tp: float, tn: float, fp: float, fn: float) -> float:
    '''
        Matthews Correlation Coefficient (MCC).
                          TP x TN - FP x FN
        MCC = --------------------------------------------
               sqrt((TP + FP)(TP + FN)(TN + FP)(TN + FN))
        Inputs:
            - tp (float): True Positives
            - tn (float): True Negatives
            - fp (float): False Positives
            - fn (float): False Negatives
        Returns:
            - MCC (float)
    '''
    return ((tp * tn) - (fp * fn)) / math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
