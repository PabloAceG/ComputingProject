from sklearn import metrics
from numpy   import matrix
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

def __parte_dataset(dataset, input_select, target_select):
    '''
        Takes a dataset and parses it to only take what is necessary.
        Inputs:
            - dataset: Input raw data as a Pandas Data Framework.
            - start: Where columns start to be useful.
        Output:
            - input: Input arrays of the dataset.
            - target: Target column of the dataset.
    '''

    num_rows    = len(dataset)
    last_column = target_select if (target_select < 0) else None

    # Input
    input_columns = dataset.columns[input_select:last_column]
    input = dataset[input_columns].head(num_rows).astype(float).to_numpy()

    # Target
    target_column = dataset.columns[target_select]
    target = dataset[target_column].head(num_rows)

    # Parse output
    sample = target[0]
    if not isinstance(sample, str): 
        target = numpy.where(target > 0, 1, 0)
    else: 
        has_faults = target == 'yes'
        target = numpy.where(has_faults , 1, 0)

    return input, target

def __data_preparation(path, input_select, target_select, type='arff', shuffle=False):
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
    data = (dataset, input_select, target_select)
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

    path       = ''     # Location of file
    start      = 0      # From which columns to use
    output_col = -2     # Position of output column
    type       = 'arff' # Type of file to be read. Default .arff

    if name == 'ant':
        # Retrieve data
        path = './dataset/ant-1.7.arff'
        start = 3

    elif name == 'apache':
        # Retrieve data
        path = './dataset/Apache.csv'
        start = 3
        output_col = -1
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

    elif name == 'hadoop-1':
        # Retrieve data
        path = './dataset/hadoop-proc-0.1.csv'
        start = 2
        output_col = 1
        type = 'csv'

    elif name == 'hadoop-2':
        # Retrieve data
        path = './dataset/hadoop-proc-0.2.csv'
        start = 2
        output_col = 1
        type = 'csv'
    
    elif name == 'hadoop-3':
        # Retrieve data
        path = './dataset/hadoop-proc-0.3.csv'
        start = 2
        output_col = 1
        type = 'csv'
    
    elif name == 'hadoop-4':
        # Retrieve data
        path = './dataset/hadoop-proc-0.4.csv'
        start = 2
        output_col = 1
        type = 'csv'
    
    elif name == 'hadoop-5':
        # Retrieve data
        path = './dataset/hadoop-proc-0.5.csv'
        start = 2
        output_col = 1
        type = 'csv'
    
    elif name == 'hadoop-6':
        # Retrieve data
        path = './dataset/hadoop-proc-0.6.csv'
        start = 2
        output_col = 1
        type = 'csv'
    
    elif name == 'hadoop-7':
        # Retrieve data
        path = './dataset/hadoop-proc-0.7.csv'
        start = 2
        output_col = 1
        type = 'csv'
    
    elif name == 'hadoop-8':
        # Retrieve data
        path = './dataset/hadoop-proc-0.8.csv'
        start = 2
        output_col = 1
        type = 'csv'
    
    
    else:
        raise Exception('The given dataset name is not valid.') 
        sys.exit(404)
    
    # Return results
    input, target = __data_preparation(path, start, output_col, type, shuffle)

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

def recall(targets, predictions) -> float:
    '''
        Sensitivity, recall, hit rate or True Positive Rate (RPR)
        https://scikit-learn.org/stable/modules/generated/sklearn.metrics.recall_score.html
               TP       TP
        TPR = ---- = --------- = 1 - FNR
               P      TP + FN
    '''
    return round(metrics.recall_score(targets, predictions), 4)

def fall_out(targets, predictions) -> float:
    '''
        Fallout, or False Positive Rate (FPR)
               FP       FP
        FPR = ---- = --------- = 1 - TNR
               N      FP + TN
    '''
    # Calculate TN and FP rates
    num_items:int = len(targets)
    false_positive:float = 0
    true_negative: float = 0
    # Count
    for (t, o) in zip(targets, predictions):
        if (o == t and o == 0):
            true_negative  += 1
        if (o != t and o == 1):
            false_positive += 1
    try:
        # Rates
        true_negative  /= num_items
        false_positive /= num_items

        # False Positive Rate
        fdr:float = false_positive / (false_positive + true_negative)
        return round(fdr, 4)
    except:
        return 0

def precision(targets, predictions) -> float:
    '''
        Precision or positive predictive value (PPV).
        https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_score.html
                 TP
        PPV = --------- = 1 - FDR
               TP + FP
    '''
    return round(metrics.precision_score(targets, predictions), 4)

def balanced(targets, predictions) -> float:
    '''
        Balanced Accuracy (BA)
        https://scikit-learn.org/stable/modules/generated/sklearn.metrics.balanced_accuracy_score.html#sklearn.metrics.balanced_accuracy_score
              TPR + TNR
        BA = -----------
                  2
    '''

    return round(metrics.balanced_accuracy_score(targets, predictions), 4)

def f1(targets, predictions) -> float:
    '''
        F1 Score. Is the harmonic mean of precision and sensitivity.
        https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html
                PPV x TPR         2 TP
        F1 = 2 ----------- = ----------------
                PPV + TPR     2 TP + FP + FN
    '''
    return round(metrics.f1_score(targets, predictions), 4)

def mcc(targets, predictions) -> float:
    '''
        Matthews Correlation Coefficient (MCC).
        https://scikit-learn.org/stable/modules/generated/sklearn.metrics.matthews_corrcoef.html
                          TP x TN - FP x FN
        MCC = --------------------------------------------
               sqrt((TP + FP)(TP + FN)(TN + FP)(TN + FN))
    '''
    return round(metrics.matthews_corrcoef(targets, predictions), 4)

def auc(targets, predictions) -> float:
    '''
        Area Under Receiver Operating Characteristic Curve, 
        Area Under ROC Curve or AUC.
        https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_curve.html
        https://scikit-learn.org/stable/modules/generated/sklearn.metrics.auc.html
               1 + TPR - FPR
        AUC = ---------------
                    2
    '''
    fpr, tpr, thresholds = metrics.roc_curve(targets, predictions)

    return round(metrics.auc(fpr, tpr), 4)

def store_results(filename:str, metrics:list):
    with open('./code/results/' + filename + '.csv', mode='a', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(metrics)

def calculate_results(targets:list, predictions:list) -> list:
    # Metrics
    ppv:float = precision(targets, predictions) 
    tpr:float = recall   (targets, predictions)
    fpr:float = fall_out (targets, predictions)
    ba :float = balanced (targets, predictions)
    fm :float = f1       (targets, predictions)
    m  :float = mcc      (targets, predictions)
    a  :float = auc      (targets, predictions)
    metrics:list  = [ppv, tpr, fpr, ba, fm, m, a]
    # Show results
    print("precision: " + str(ppv))
    print("recall: "    + str(tpr))
    print("fall_out: "  + str(fpr))
    print("balanced: "  + str(ba))
    print("f1: "        + str(fm))
    print("mcc: "       + str(m))
    print("auc: "       + str(a))

    return metrics

def train_predict(clf, input_train, target_train, test):
    # Train
    clf.fit(input_train, target_train)
    # Test
    return clf.predict(test)
