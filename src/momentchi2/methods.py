from scipy.stats import gamma
from scipy.stats import f
import numpy as np
import warnings

def addOne(x):
    return(x+1)


# Hall-Buckley-Eagleson
def hbe(coeff, x):
    '''Hall-Buckley-Eagleson method

    Computes the cdf of a positively-weighted sum of chi-squared
    random variables with the Hall-Buckley-Eagleson (HBE) method.


    Parameters:
    coeff (list or numpy array): The coefficient vector. 
                                 All values must be greater than 0.

    x (list or numpy array or float): The vector of quantile values. 
                                      All values must be greater than 0.


    Returns:
    The cdf of the x value(s). It is returned as the same type as x, 
    i.e. if x is a list, it is returned as a list; if x is a numpy array
    it is returned as a numpy array.


    Details:
     * Note that division assumes Python 3, so may not work with Python 2.
     * Depends on numpy libary for the arrays.
     * If lists are passed, they are converted to numpy arrays (and back again).
     * Depends on scipy library for scipy.stats.gamma function.


    Examples:
    #Examples taken from Table 18.6 in N. L. Johnson, S. Kotz, N. Balakrishnan.
    #Continuous Univariate Distributions, Volume 1, John Wiley & Sons, 1994.

    # how to load the hbe function from momenthchi2
    from momentchi2.methods import hbe

    # should give value close to 0.95, actually 0.94908
    hbe([1.5, 1.5, 0.5, 0.5], 10.203)            

    # should give value close to 0.05, but is actually 0.02857
    hbe([1.5, 1.5, 0.5, 0.5], 0.627)            

    # specifying parameters
    hbe(coeff=[1.5, 1.5, 0.5, 0.5], x=10.203)            

    # x is a list, output approx. 0.05, 0.95
    hbe([1.5, 1.5, 0.5, 0.5], [0.627, 10.203])  

    # instead of lists can be numpy arrays 
    # (any list is converted to a numpy arrays inside the function anyway)
    import numpy as np
    hbe( np.array([1.5, 1.5, 0.5, 0.5]), np.array([0.627, 10.203]) )  


    References:
     * P. Hall. Chi squared approximations to the distribution of a
       sum of independent random variables. The Annals of
       Probability, 11(4):1028-1036, 1983.

     * M. J. Buckley and G. K. Eagleson. An approximation to the
       distribution of quadratic forms in normal random variables.
       Australian Journal of Statistics, 30(1):150-159, 1988.

     * D. A. Bodenham and N. M. Adams. A comparison of efficient 
       approximations for a weighted sum of chi-squared random variables. 
       Statistics and Computing, 26(4):917-928, 2016.
    '''
    # some checking, so that passing lists/arrays does not matter
    if isinstance(coeff, list):
        coeff = np.array(coeff)

    isList = False
    if not isinstance(x, float):
        if isinstance(x, list):
            isList = True
            x = np.array(x)

    # the next two lines work for floats or numpy arrays, but not lists
    K_1 = sum(coeff)
    K_2 = 2 * sum(coeff**2)
    K_3 = 8 * sum(coeff**3)
    nu = 8 * (K_2**3) / (K_3**2)
    k = nu / 2
    theta = 2
    # in the next line x can be a float or numpy array, but not a list
    x = (  (2 * nu / K_2)**(0.5)  ) * (x - K_1) + nu 
    p = gamma.cdf(x, a=k, scale=theta)

    # if x was passed as a list, will return a list
    if isList:
        p = p.tolist()
    return(p)



# Satterthwaite-Welch
def sw(coeff, x):
    '''Satterthwaite-Welch method

    Computes the cdf of a positively-weighted sum of chi-squared
    random variables with the Satterthwaite-Welch (SW) method.


    Parameters:
    coeff (list or numpy array): The coefficient vector. 
                                 All values must be greater than 0.

    x (list or numpy array or float): The vector of quantile values. 
                                      All values must be greater than 0.


    Returns:
    The cdf of the x value(s). It is returned as the same type as x, 
    i.e. if x is a list, it is returned as a list; if x is a numpy array
    it is returned as a numpy array.


    Details:
     * Note that division assumes Python 3, so may not work with Python 2.
     * Depends on numpy libary for the arrays.
     * If lists are passed, they are converted to numpy arrays (and back again).
     * Depends on scipy library for scipy.stats.gamma function.


    Examples:
    #Examples taken from Table 18.6 in N. L. Johnson, S. Kotz, N. Balakrishnan.
    #Continuous Univariate Distributions, Volume 1, John Wiley & Sons, 1994.

    # how to load the sw function from momenthchi2
    from momentchi2.methods import sw

    # should give value close to 0.95, actually 0.95008
    sw([1.5, 1.5, 0.5, 0.5], 10.203)            

    # should give value close to 0.05, but is actually 0.0657
    sw([1.5, 1.5, 0.5, 0.5], 0.627)            

    # specifying parameters
    sw(coeff=[1.5, 1.5, 0.5, 0.5], x=10.203)            

    # x is a list, output approx. 0.05, 0.95
    sw([1.5, 1.5, 0.5, 0.5], [0.627, 10.203])  

    # instead of lists can be numpy arrays 
    # (any list is converted to a numpy arrays inside the function anyway)
    import numpy as np
    sw( np.array([1.5, 1.5, 0.5, 0.5]), np.array([0.627, 10.203]) )  


    References:
     * B. L.Welch. The significance of the difference between two
       means when the population variances are unequal.
       Biometrika, 29(3/4):350-362, 1938.

     * F. E. Satterthwaite. An approximate distribution of estimates
       of variance components. Biometrics Bulletin, 2(6):110-114,

     * G. E. P. Box Some theorems on quadratic forms applied in the
       study of analysis of variance problems, I. Effects of
       inequality of variance in the one-way classification. _The
       Annals of Mathematical Statistics_, 25(2):290-302, 1954.

     * D. A. Bodenham and N. M. Adams. A comparison of efficient 
       approximations for a weighted sum of chi-squared random variables. 
       Statistics and Computing, 26(4):917-928, 2016.
    '''
    # some checking, so that passing lists/arrays does not matter
    if isinstance(coeff, list):
        coeff = np.array(coeff)

    isList = False
    if not isinstance(x, float):
        if isinstance(x, list):
            isList = True
            x = np.array(x)

    w = sum(coeff)
    u = sum(coeff**2) / (w**2)
    k = 0.5 / u
    theta = 2 * u * w
    p = gamma.cdf(x, a=k, scale=theta)

    # if x was passed as a list, will return a list
    if isList:
        p = p.tolist()
    return(p)




# Wood's F method
def wf(coeff, x):
    '''Wood's F method

    Computes the cdf of a positively-weighted sum of chi-squared
    random variables with the Wood F (WF) method.

    Parameters:
    coeff (list or numpy array): The coefficient vector. 
                                 All values must be greater than 0.

    x (list or numpy array or float): The vector of quantile values. 
                                      All values must be greater than 0.


    Returns:
    The cdf of the x value(s). It is returned as the same type as x, 
    i.e. if x is a list, it is returned as a list; if x is a numpy array
    it is returned as a numpy array.


    Details:
     * Note that division assumes Python 3, so may not work with Python 2.
     * Depends on numpy libary for the arrays.
     * If lists are passed, they are converted to numpy arrays (and back again).
     * Depends on scipy library for scipy.stats.f function.

    Note:
    There are pathological cases where, for certain
    coefficient vectors (which result in certain cumulant values), the
    Wood F method will be unable to match moments (cumulants) with the
    three-parameter _F_ distribution. In this situation, the HBE
    method is used, and a warning is displayed. A simple example of
    such a pathological case is when the coefficient vector is of
    length 1. Note that these pathological cases are rare; see (Wood,
    1989) in the references.


    Examples:
    #Examples taken from Table 18.6 in N. L. Johnson, S. Kotz, N. Balakrishnan.
    #Continuous Univariate Distributions, Volume 1, John Wiley & Sons, 1994.

    # how to load the wf function from momenthchi2
    from momentchi2.methods import wf

    # should give value close to 0.95, actually 0.951058
    wf([1.5, 1.5, 0.5, 0.5], 10.203)            

    # should give value close to 0.05, but is actually 0.055739
    wf([1.5, 1.5, 0.5, 0.5], 0.627)            

    # specifying parameters
    wf(coeff=[1.5, 1.5, 0.5, 0.5], x=10.203)            

    # x is a list, output approx. 0.05, 0.95
    wf([1.5, 1.5, 0.5, 0.5], [0.627, 10.203])  

    # instead of lists can be numpy arrays 
    # (any list is converted to a numpy arrays inside the function anyway)
    import numpy as np
    wf( np.array([1.5, 1.5, 0.5, 0.5]), np.array([0.627, 10.203]) )  


    # Pathological case; throws warning and calls hbe() method
    wf([0.9], 1) 

    References:

     * A. T. A. Wood. An F approximation to the distribution of a
       linear combination of chi-squared variables. Communications
       in Statistics-Simulation and Computation, 18(4):1439-1456,
       1989.

     * D. A. Bodenham and N. M. Adams. A comparison of efficient 
       approximations for a weighted sum of chi-squared random variables. 
       Statistics and Computing, 26(4):917-928, 2016.
    '''
    # some checking, so that passing lists/arrays does not matter
    if isinstance(coeff, list):
        coeff = np.array(coeff)

    isList = False
    if not isinstance(x, float):
        if isinstance(x, list):
            isList = True
            x = np.array(x)

    # the next two lines work for floats or numpy arrays, but not lists
    K_1 = sum(coeff)
    K_2 = 2 * sum(coeff**2)
    K_3 = 8 * sum(coeff**3)
    r_1 = 4 * ( K_2**2 ) * K_1 + K_3 * ( K_2 - (K_1**2) )
    r_2 = K_3 * K_1 - 2 * (K_2**2)

    # check for pathological case, else call WF method
    if (r_1==0) or (r_2==0):
        warnings.warn("Pathological case: either r1 or r2 equals 0: running hbe instead.")
        p = hbe(coeff, x)
    else:
        beta = r_1 / r_2
        alpha_1 = 2 * K_1 * ( K_3 * K_1 + (K_1**2) * K_2 - (K_2**2) )/r_1
        alpha_2 = 3 + 2 * K_2 * ( K_2 + (K_1**2) )/r_2
        x = x * alpha_2/(alpha_1 * beta)
        p = f.cdf(x, dfn = 2 * alpha_1, dfd = 2 * alpha_2)

    # if x was passed as a list, will return a list
    if isList:
        p = p.tolist()
    return(p)
