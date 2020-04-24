import pyRserve
import sys

'''
    Connects to an R server specified by user.
    The function accepts parameters for configuration.
'''
class r_connect:

    __metrics = None

    def __init__ (self):
        self.__connection = self.__connect()
        
    '''
        Create connection to R's RPC.
    '''
    def __connect (self):
        return pyRserve.connect()

    '''
        Obtains the metrics for a given dataset.
    '''
    def get_metrics (self, X=None, Y=None):
        if X is None and Y is None :
            if self.__metrics is not None :
                return self.__metrics
            else :
                # No data and no parameters: finish execution
                error_message = '''
                    No metrics so far! Try given the dataset vector and 
                    target vector as parameters.\n
                '''
                raise Exception(error_message)
                sys.exit (400)
        else :
            # Stores connection to R's RPC.
            connect = self.__connection

            # Sends the input matrix and the output vector to R.
            connect.r.X = X
            connect.r.y = Y
            
            # Library to use in R.
            connect.r('df_X <- as.data.frame(X)')
            connect.r('df_y <- as.data.frame(y)')
            connect.r('library("ECoL")')
            
            ## Metrics, uses a dictionary to provide a faster access to its 
            # contents.
            metrics = []

            # Balance
            balance = connect.r('balance(df_X, df_y)') 
            for i in range(len(balance)):
                metrics.append(balance[i])
            message = '# Balance (C1, C2):\t'
            balance_dic_entry = {'balance' : [message, balance] }
            print(message, balance)

            # Correlation
            correlation = connect.r('correlation(df_X, df_y)')
            for i in range(len(correlation)):
                metrics.append(correlation[i])
            print('# Correlation (C1, C2, C3, C4):\t', correlation)

            # Dimensionality
            dimensionality = connect.r('dimensionality(df_X, df_y)')
            for i in range(len(dimensionality)):
                metrics.append(dimensionality[i])
            print('# Dimensionality (T2, T3, T4):', dimensionality)

            # Linearity
            linearity = connect.r('linearity.class(df_X, df_y)')
            for i in range(len(linearity)):
                metrics.append(linearity[i])
            print('# Linearity (L1, L2, L3):\t', linearity)

            # Neighborhood
            neighborhood = connect.r('neighborhood(df_X, df_y)')
            for i in range(len(neighborhood)):
                metrics.append(neighborhood[i])
            print('# Neighborhood (N1, N2, N3, N4, T1, LSC):\t', neighborhood)

            # Network
            network = connect.r('network(df_X, df_y)')
            for i in range(len(network)):
                metrics.append(network[i])
            print('# Network (Density, ClsCoef, Hubs):\t', network)

            # Overlap
            overlap = connect.r('overlapping(df_X, df_y)')
            for i in range(len(overlap)):
                metrics.append(overlap[i])
            print('# Overlap (F1, F1v, F2, F3, F4):\t', overlap)

            # Smoothness
            smoothness = connect.r('smoothness(df_X, df_y)')
            for i in range(len(smoothness)):
                metrics.append(smoothness[i])
            print('# Smoothness (S1, S2, S3, S4):\t', smoothness)
            
            self.__metrics = metrics

            return metrics

    '''
        Print the metrics previously calculated.
    '''
    def print_metrics (self, metrics=None) :
        if metrics is None:
            if self.__metrics is not None : 
                metrics = self.__metrics
            else :
                # No data and no parameters: finish execution
                error_message = '''
                    No metrics are stored or passed as parameter. Try to 
                    generate them first -> getMetrics(data, target)\n
                '''
                raise Exception (error_message)
                sys.exit (400)

        for m in metrics :
            print(m)


    '''
        Both calculates the metrics and prints them along with a few some
        debug messages.
    '''
    def get_print_metrics(self, X, Y):
        print ('Calculating metrics for the specified dataset.', end='\n')
        # Obtain metrics back
        self.get_metrics (X, Y)
        print ('Metrics calculated, starting to print.', end='\n')

        # Print results
        self.print_metrics (self.__metrics)
