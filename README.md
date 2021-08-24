# momentchi2py

A Python implementation of three methods for computing the 
cumulative distribution function of a weighted sum of chi-squared
random variables.

## Methods

Based on the R package `momentchi2`, this Python version contains the following methods:
  - Hall-Buckley-Eagleson (function `hbe`)
  - Satterthwaite-Welch (function `sw`)
  - Wood's F method (function `wf`)

The Lindsay-Pilla-Basak method will be added later.

## Installation instructions

The package is in the process of adding the package to PyPi, but 
the `methods.py` script in the `src/momentchi2` folder is all you need. 
That script which contains the functions `hbe`, `sw` and `wf`

## Package dependencies

The package needs `numpy` and `scipy` are required.


## Which method should I use?

All three methods are good, but the Hall-Buckley-Eagleson method
is recommended for most situations. See Bodenham and Adams (2016)
for a detailed analysis.


## Examples:

```
## Hall-Buckley-Eagleson method
# how to load the hbe function from momenthchi2
from momentchi2.methods import hbe

# should give value close to 0.95, actually 0.94908
hbe([1.5, 1.5, 0.5, 0.5], 10.203)            

# x is a list, output approx. 0.05, 0.95
hbe([1.5, 1.5, 0.5, 0.5], [0.627, 10.203])  

# x is a numpy array - preferred approach for speed
import numpy as np
from momentchi2.methods import hbe
hbe( np.array([1.5, 1.5, 0.5, 0.5]), np.array([0.627, 10.203]) )  
```

## Details

Input for quantile vector x can  lists or numpy arrays, but 

### Package references

 1. D. A. Bodenham and N. M. Adams. A comparison of efficient 
   approximations for a weighted sum of chi-squared random variables. 
   Statistics and Computing, 26(4):917-928, 2016.

 2. D. A. Bodenham (2016). momentChi2: Moment-Matching Methods for Weighted Sums of Chi-Squared 
   Random Variables, [https://cran.r-project.org/package=momentchi2](https://cran.r-project.org/package=momentchi2)


### Method references

#### Satterthwaite-Welch

 3. B. L.Welch. The significance of the difference between two
    means when the population variances are unequal.
    Biometrika, 29(3/4):350-362, 1938.

 4. F. E. Satterthwaite. An approximate distribution of estimates
    of variance components. Biometrics Bulletin, 2(6):110-114,

 5. G. E. P. Box Some theorems on quadratic forms applied in the
    study of analysis of variance problems, I. Effects of
    inequality of variance in the one-way classification. _The
    Annals of Mathematical Statistics_, 25(2):290-302, 1954.


#### Hall-Buckley-Eagleson

 6. P. Hall. Chi squared approximations to the distribution of a
    sum of independent random variables. The Annals of
    Probability, 11(4):1028-1036, 1983.

 7. M. J. Buckley and G. K. Eagleson. An approximation to the
    distribution of quadratic forms in normal random variables.
    Australian Journal of Statistics, 30(1):150-159, 1988.


#### Wood's F method

  8. A. T. A. Wood. An F approximation to the distribution of a
     linear combination of chi-squared variables. Communications
     in Statistics-Simulation and Computation, 18(4):1439-1456,
     1989.
