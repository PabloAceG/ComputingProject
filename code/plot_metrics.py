import os

import pandas  as pd
import seaborn as sns
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # Path to files storing metrics
    results_path = './code/results/'

    # Retreive metrics
    metrics_apache = pd.read_csv(results_path + 'metrics-apache.csv')
    metrics_hadoop = pd.read_csv(results_path + 'metrics-hadoop.csv')
    labels  = metrics_apache['class'] + '-' + metrics_apache['metric']
    metrics_apache.drop(columns=['class', 'metric'], inplace=True)
    metrics_hadoop.drop(columns=['class', 'metric'], inplace=True)

    columns = metrics_apache.columns.tolist() + metrics_hadoop.columns.tolist()

    num_items = len(labels)
    for item in range(num_items):
        values  = pd.concat([metrics_apache.iloc[item], metrics_hadoop.iloc[item]])
        fig, ax = plt.subplots(figsize=(20, 4))
        plt.scatter(columns, values)

        num_datasets = len(metrics_apache.columns)
        for n in range(num_datasets):
            xy = (metrics_apache.columns[n], values[n])
            ax.annotate(values[n], xy=xy, textcoords='data')

        num_datasets = len(metrics_hadoop.columns)
        for n in range(num_datasets):
            xy = (metrics_hadoop.columns[n], values[n])
            ax.annotate(values[n], xy=xy, textcoords='data')

        plt.savefig('./code/figures/' + labels[item] + '.png', 
                    bbox_inches='tight')
    

