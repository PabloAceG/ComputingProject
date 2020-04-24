## Implementation

### Requirements
- R-Studio installed.
- Python 3.x installed.
- [ECol](https://cran.r-project.org/web/packages/ECoL/index.html) package (for 
R): Metrics.
- [Rserve](https://www.rforge.net/Rserve/) package (for R): RPC (Remote 
Procedural Call) server.
- [pyRserve](https://pythonhosted.org/pyRserve/) library (for Python): to 
connect with Rserve.
- `r_connect` file (for Python): Connection implementation between `Rserve` and
`pyRserve` that allows to obtain the metrics from a dataset.

### Disclaimer
Thanks to [R012](https://github.com/R012) for developing the connector between R and Python, which allows 
to use for example the [ECoL Library](https://github.com/lpfgarcia/ECoL) for 
pattern recognition and complexity-measures.
