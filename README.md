# Introduction

This repository contains my dissertation. I presented this work as my final
year project. 

The paper is about how datasets are affected by the values of complexity 
metrics, and how some techniques that try to mitigate the effect of some of 
those metrics affect the evaluation results.

# Pre-requisites

To be able to execute the experiments within the repository, Python3 is needed. 
Anaconda or the official Python located in the official repositories can be used 
as long as version 3 or posterior is used. Trying to replicate the experiments 
on some operative systems might terminate in error. If this is the case, python
can be changed (_Linux_) with:

```bash
sudo update-alternatives --config python
```

R (programming language) is needed before trying to execute the project. Also, 
the following packages are mandatory in order to replicate the experiments:


- ECoL - Dataset Complexity Metrics Package.
  - [ECoL GitHub](https://github.com/lpfgarcia/ECoL)
  - [ECoL Documentation](https://cran.r-project.org/web/packages/ECoL/)
- [Rserve](https://rforge.net/Rserve/doc.html) - server, responds requests 
made to R.


Once the previous requisites are fulfilled, the R server can be started by 
executing the following commands:

```python
library(Rserve) # import the library
run.Rserve() # start the server. Or simply Rserve()
``` 

Now, it is time to download the project to install the remaining Python 
packages. The project can be downloaded from 
[this repo](https://github.com/PabloAceG/ComputingProject/).

Same as before, some Python packages are mandatory to execute the project. These
packages are available in 
[requirements.txt](https://github.com/PabloAceG/ComputingProject/blob/master/code/requirements.txt)
file. To automatically install those packages, run (execute all commands from
parent repository):

```bash
pip install -r .\code\requirements.txt
```

It might happen that `pip install -r` might not install all packages. 
To solve this, the failing packages must be installed manually:

```bash
pip install <package_name>
```

# Execution

Now, the experiments should be replicable. The experiment's code is under the 
[code folder](https://github.com/PabloAceG/ComputingProject/tree/master/code).
To run them, execute: 

```bash
python code/metrics_comparison.py
python code/metrics_kfold.py
python code/metrics_kfold_undersampling.py
python code/metrics_kfold_oversampling.py
```

Each of the previous commands execute one experiment.

As final remarks, the class `r_connect.py` (go 
[here](https://github.com/PabloAceG/ComputingProject/blob/master/code/r_connect.py))
is the client connection the server in R (Rserve). It makes the requests to the 
`ECoL`package to obtain the complexity metrics.

The class `data.py` (go
[here](https://github.com/PabloAceG/ComputingProject/blob/master/code/data.py))
standardizes the datasets input (parsing data) and some other metrics from the 
package [sklearn](https://scikit-learn.org/stable/index.html).