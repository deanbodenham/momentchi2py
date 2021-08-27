# momentchi2

A Python implementation of three approximate methods for computing the 
cumulative distribution function of a weighted sum of chi-squared
random variables. All the methods are based on moment-matching techniques.

## Methods

Based on the R package `momentchi2`, this Python version contains the following methods:
  - Hall-Buckley-Eagleson (function `hbe`)
  - Satterthwaite-Welch (function `sw`)
  - Wood's F method (function `wf`)
  - Lindsay-Pilla-Basak method (function `lpb4`)


## Installation instructions

Install using `pip`:

```
python3 -m pip install momentchi2
```

## Package dependencies

The packages `numpy` and `scipy` are required to be installed.


## Which method should I use?

All four methods (`sw`, `hbe`, `wf` and `lpb4`) are good, 
but the **Hall-Buckley-Eagleson** method is recommended for situations 
where the number of coefficients is modertately large 
(say, greater than 100). For a smaller number of coefficients (e.g. up to 10), 
the Lindsay-Pilla-Basak method is recommended.
See Bodenham and Adams (2016) for a detailed analysis.



## Examples:

```
## Hall-Buckley-Eagleson method
from momentchi2 import hbe

# should give value close to 0.95, actually 0.94908
hbe(coeff=[1.5, 1.5, 0.5, 0.5], x=10.203)            

# x is a list, output approx. 0.05, 0.95
hbe([1.5, 1.5, 0.5, 0.5], [0.627, 10.203])  

# x is a numpy array - preferred approach for speed
import numpy as np
from momentchi2 import hbe
hbe( np.array([1.5, 1.5, 0.5, 0.5]), np.array([0.627, 10.203]) )  

# Other methods, e.g. sw, wf or lpb4
# All methods called: methodname(coeff, x)
from momentchi2 import sw
sw([1.5, 1.5, 0.5, 0.5], [0.627, 10.203])  

from momentchi2 import wf
wf([1.5, 1.5, 0.5, 0.5], [0.627, 10.203])  

from momentchi2 import lpb4
lpb4([1.5, 1.5, 0.5, 0.5], [0.627, 10.203])  

# for a larger number of coefficients in coeff vector, 
# can increase the number of moments p for improved accuracy.
# NOTE: we need len(coeff) >= p. Default value of p is p=4.
lpb4([0.1, 2.3, 3.4, 5.6, 7.8, 8.9, 9.1], [9.366844, 82.0018], p=6)  
```

## Details


All methods take two input arguments:
  * `coeff`: a list of the coefficients of the weighted sum 
  (where all values must be strictly greater than 0), and  
  * `x`: the quantile value(s) at which point(s) the cumulative distribution
  function is computed.  

So calling a method is: `methodname(coeff, x)`, where e.g. `methodname` is `hbe`.

Input for quantile vector `x` can be a float (single value) or a list of values, 
or a numpy array. Internally, lists are converted to numpy arrays (and then back
to lists), so that the output format of `x` is the same as the input format.

The Lindsay-Pilla-Basak (`lpb4`) method has a parameter `p` which is set
to 4 by default and this is sufficient in most cases. 
If the number of coefficients is larger (e.g. greater than 8), then
the `lpb4` method can be used for larger . Of course, the increased accuracy 
comes at an increased computational cost.

There are a few pathological cases where Wood's F method or the 
Lindsay-Pilla-Basak method can fail (e.g. number of coefficients < p), 
in which case the `hbe` method will be called.


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

#### Lindsay-Pilla-Basak method

  9. B. G. Lindsay, R. S. Pilla, and P. Basak. Moment-based
     approximations of distributions using mixtures: Theory and
     applications. Annals of the Institute of Statistical Mathematics, 
     52(2):215-230, 2000.


### An exact solution: Imhof's method

Note that while these methods are all approximate, they are very fast and
are accurate to two or three decimal places.  If an 
exact answer is required to arbitrary accuracy, consider Imhof's method, which 
is implemented in the R package `CompQuadForm`.

  10. J. P. Imhof. Computing the distribution of quadratic forms in normal 
      variables. Biometrika 48(3/4): 419-426, 1961.


