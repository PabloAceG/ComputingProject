# Libraries
from r_connect import r_connect
from numpy     import matrix
import numpy
import math
import data              as DATASETS
import matplotlib.pyplot as plt

def add_metrics(storage, dataset_metrics):
    '''
        Includes a new set of calculated metrics into a variable.
        It creates an array for each metric's measure.
        Input:
            - storage(dict) : Data structure to store the list of measures from 
                the metrics of datasets.
            - dataset_metrics (dict): Metrics computed from a dataset.
        Output:
            - storage (dict): Metrics added to the already stored ones.
    '''
    # Iterate through the metrics
    for m in storage:
        # Metric might not have been calculated for dataset
        if m in dataset_metrics:
            metric = dataset_metrics[m]
      
            # Iterate through the measures of the metric
            for measure in storage[m]:
                try:
                    # Measure might not have been calculated for dataset
                    if measure in storage[m]:
                        storage[m][measure].append(float(metric[measure]))
                except Exception:
                    pass
        else:
            print('Something went wrong')

    return storage

def plot_metrics_comparison(metrics):
    '''
        Input:
            - metrics:
        Output:
    '''
    i = 0
    j = 0
    for metric in results:
        rows = 2
        cols = int(math.ceil(len(results[metric]) / rows))
        fig, axs = plt.subplots(rows, cols)
        for measure in results[metric]:
            if cols == 1:
                if results[metric][measure]:
                    axs[i].bar(datasets, results[metric][measure])
                
                else:
                    axs[i].text(0.5, 0.5, 'No data')
            
                title = metric + ': ' + measure
                axs[i].set_title(title)
            else:
                if results[metric][measure]:
                    axs[i, j].bar(datasets, results[metric][measure])
                
                else:
                    axs[i, j].text(0.5, 0.5, 'No data')
                
                title = metric + ': ' + measure
                axs[i, j].set_title(title)
            
            if j == cols - 1:
                i += 1
                j =  0
            else:
                j += 1
            
        i = 0
        j = 0
        plt.show()   


if __name__ == '__main__':
    '''
        Main code - to be executed
    '''

    # Datasets to analyze
    datasets = [
        'ant', 
        'apache',
        'camel',
        'ivy',
        'jedit',
        'log4j',
        'synapse',
        'xalan',
        'xerces'
    ]

    # Store metrics computed from datasets (datatype: dict)
    results = {
        'balance': {
            'C1': [],
            'C2': []
        },
        'correlation': {
            'C1': [],
            'C2': [],
            'C3': [],
            'C4': []
        },
        'dimensionality': {
            'T1': [],
            'T2': [],
            'T3': []
        },
        'linearity': {
            'L1': [],
            'L2': [],
            'L3': []
        },
        'neighborhood': {
            'N1':  [],
            'N2':  [],
            'N3':  [],
            'N4':  [],
            'T1':  [],
            'LSC': [],
        },
        'network': {
            'Density': [],
            'ClsCoef': [],
            'Hubs':    []
        },
        'overlap': {
            'F1':  [],
            'F1v': [],
            'F2':  [],
            'F3':  [],
            'F4':  []
        },
        'smoothness': {
            'S1': [],
            'S2': [],
            'S3': [],
            'S4': []
        }
    }
    
    # Connect to R
    connector = r_connect()

    # Get Metrics
    for set in datasets:
        print(set)
        inputs, targets = DATASETS.get_dataset(set)
        metrics = connector.get_metrics(inputs, targets)

        results = add_metrics(results, metrics)

    # Print Metrics
    print(results)
    plot_metrics_comparison(results)
 