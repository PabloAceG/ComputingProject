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
            metrics = {}

            # Balance
            balance = self.safe_connect('balance(df_X, df_y)') 
            message = '# Balance (C1, C2):\t'
            balance_dic_entry = { 'balance' : [message, balance] }
            metrics.update (balance_dic_entry)

            # Correlation
            correlation = self.safe_connect('correlation(df_X, df_y)')
            message = '# Correlation (C1, C2, C3, C4):\t'
            correlation_dic_entry = { 'correlation' : [message, correlation] } 
            metrics.update (correlation_dic_entry)

            # Dimensionality
            dimensionality = self.safe_connect('dimensionality(df_X, df_y)')
            message = '# Dimensionality (T2, T3, T4):'
            dimensionality_dic_entry = { 'dimensionality' : [message, dimensionality] }
            metrics.update (dimensionality_dic_entry)

            # Linearity
            linearity = self.safe_connect('linearity(df_X, df_y)')
            message = '# Linearity (L1, L2, L3):\t'
            linearity_dic_entry = { 'linearity' : [message, linearity] }
            metrics.update (linearity_dic_entry)

            # Neighborhood
            neighborhood = self.safe_connect('neighborhood(df_X, df_y)')
            message = '# Neighborhood (N1, N2, N3, N4, T1, LSC):\t'
            neighborhood_dic_entry = { 'neighborhood' : [message, neighborhood] }
            metrics.update (neighborhood_dic_entry)

            # Network
            network = self.safe_connect('network(df_X, df_y)')
            message = '# Network (Density, ClsCoef, Hubs):\t'
            network_dic_entry = { 'network' : [message, network] }
            metrics.update (network_dic_entry)

            # Overlap
            overlap = self.safe_connect('overlapping(df_X, df_y)')
            message = '# Overlap (F1, F1v, F2, F3, F4):\t'
            overlap_dic_entry = { 'overlap' : [message, overlap] }
            metrics.update (overlap_dic_entry)

            # Smoothness
            smoothness = self.safe_connect('smoothness(df_X, df_y)')
            message = '# Smoothness (S1, S2, S3, S4):\t'
            smoothness_dic_entry = { 'smoothness' : [message, smoothness] }
            metrics.update (smoothness_dic_entry)
            
            self.__metrics = metrics

            return metrics

    '''
        Print the metrics previously calculated.
    '''
    def print_metrics (self, metrics=None) :

        print ('\n\n=== Printing metrics ===', end='\n\n')

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
        
        # Positions within the components of the dictionart
        header  = 0
        results = 1
                
        # In dictionary
        # Name = Key
        # Components = Values
        for name, components in metrics.items() :
            print(components[header], components[results])

    '''
        Both calculates the metrics and prints them along with a few some
        debug messages.
    '''
    def get_print_metrics(self, X, Y):
        print ('\n\n=== Calculating metrics for the specified dataset ===', end='\n\n')
        
        # Obtain metrics back
        self.get_metrics (X, Y)
        print ('\n\n=== Metrics calculated, starting to print ===', end='\n')

        # Print results
        self.print_metrics (self.__metrics)

        return self.__metrics

    '''
        Perform a safe connection to R, with error treatment.
    '''
    def safe_connect(self, operation) :
        connection = self.__connection
        
        return connection.r(operation)
        """
        try:
            metric = connection.r(operation)
        except Exception as err_msg:
            metric = None
            print ('Could not retrieve {0}!: {0}'.format (operation, err_msg) )
        finally :
            return metric
        """
        
