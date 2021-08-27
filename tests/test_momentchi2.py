import unittest
import numpy as np

from momentchi2 import hbe
from momentchi2 import sw
from momentchi2 import wf
from momentchi2 import lpb4

from momentchi2.utilities import sum_of_powers
from momentchi2.utilities import update_moments
from momentchi2.utilities import get_moments_from_cumulants
from momentchi2.utilities import get_cumulant_vec_vectorised
from momentchi2.utilities import get_weighted_sum_of_chi_squared_moments
from momentchi2.utilities import get_lambdatilde_1
from momentchi2.utilities import deltaNmat_applied
from momentchi2.utilities import det_deltaNmat
from momentchi2.utilities import get_lambdatilde_p
from momentchi2.utilities import get_base_vector
from momentchi2.utilities import get_ith_coeff_of_Stilde_poly
from momentchi2.utilities import get_Stilde_poly_coeff
from momentchi2.utilities import get_VDM_b_vec
from momentchi2.utilities import get_vandermonde
from momentchi2.utilities import get_real_poly_roots
from momentchi2.utilities import gen_and_solve_VDM_system
from momentchi2.utilities import get_mixed_pval_vec

from momentchi2.utilities import checkCoeffAllPositive
from momentchi2.utilities import checkXAllPositive
from momentchi2.utilities import getCoeffError
from momentchi2.utilities import getXError



class BasicTests(unittest.TestCase):

    def test_basic1(self):
        '''Testing if two numbers are equal, just to get started
        '''
        x = 0
        y = 0
        self.assertEqual(x, y)

class CoeffAndXTests(unittest.TestCase):

    def test_checkCoeffAllPositive1(self):
        '''checkCoeffAllPositive int > 0
        '''
        coeff = 3
        soln = True
        ans = checkCoeffAllPositive(coeff)
        self.assertEqual(ans, soln)


    def test_checkCoeffAllPositive2(self):
        '''checkCoeffAllPositive int = 0
        '''
        coeff = 0
        soln = False
        ans = checkCoeffAllPositive(coeff)
        self.assertEqual(ans, soln)


    def test_checkCoeffAllPositive3(self):
        '''checkCoeffAllPositive int < 0
        '''
        coeff = -1
        soln = False
        ans = checkCoeffAllPositive(coeff)
        self.assertEqual(ans, soln)


    def test_checkCoeffAllPositive4(self):
        '''checkCoeffAllPositive float < 0
        '''
        coeff = -0.5
        soln = False
        ans = checkCoeffAllPositive(coeff)
        self.assertEqual(ans, soln)


    def test_checkCoeffAllPositive5(self):
        '''checkCoeffAllPositive float > 0
        '''
        coeff = 0.5
        soln = True
        ans = checkCoeffAllPositive(coeff)
        self.assertEqual(ans, soln)


    def test_checkCoeffAllPositive6(self):
        '''checkCoeffAllPositive list > 0
        '''
        coeff = [0.5, 1]
        soln = True
        ans = checkCoeffAllPositive(coeff)
        self.assertEqual(ans, soln)


    def test_checkCoeffAllPositive7(self):
        '''checkCoeffAllPositive list one = 0
        '''
        coeff = [0.5, 0, 1.2]
        soln = False
        ans = checkCoeffAllPositive(coeff)
        self.assertEqual(ans, soln)


    def test_checkCoeffAllPositive8(self):
        '''checkCoeffAllPositive list one < 0
        '''
        coeff = [0.5, 1.5, -1.2]
        soln = False
        ans = checkCoeffAllPositive(coeff)
        self.assertEqual(ans, soln)


    def test_checkXAllPositive1(self):
        '''checkXAllPositive int > 0
        '''
        x = 3
        soln = True
        ans = checkXAllPositive(x)
        self.assertEqual(ans, soln)


    def test_checkXAllPositive2(self):
        '''checkXAllPositive int = 0
        '''
        x = 0
        soln = False
        ans = checkXAllPositive(x)
        self.assertEqual(ans, soln)


    def test_checkXAllPositive3(self):
        '''checkXAllPositive int < 0
        '''
        x = -1
        soln = False
        ans = checkXAllPositive(x)
        self.assertEqual(ans, soln)


    def test_checkXAllPositive4(self):
        '''checkXAllPositive float < 0
        '''
        x = -0.5
        soln = False
        ans = checkXAllPositive(x)
        self.assertEqual(ans, soln)


    def test_checkXAllPositive5(self):
        '''checkXAllPositive float > 0
        '''
        x = 0.5
        soln = True
        ans = checkXAllPositive(x)
        self.assertEqual(ans, soln)


    def test_checkXAllPositive6(self):
        '''checkXAllPositive list > 0
        '''
        x = [0.5, 1]
        soln = True
        ans = checkXAllPositive(x)
        self.assertEqual(ans, soln)


    def test_checkXAllPositive7(self):
        '''checkXAllPositive list one = 0
        '''
        x = [0.5, 0, 1.2]
        soln = False
        ans = checkXAllPositive(x)
        self.assertEqual(ans, soln)


    def test_checkXAllPositive8(self):
        '''checkXAllPositive list one < 0
        '''
        x = [0.5, 1.5, -1.2]
        soln = False
        ans = checkXAllPositive(x)
        self.assertEqual(ans, soln)


    def test_getXError1(self):
        '''getXError list one = 0
        '''
        x = [0.5, 0, 1.2]
        soln = "all values in x need to be strictly > 0."
        ans = getXError(x)
        self.assertEqual(ans, soln)


    def test_getXError2(self):
        '''getXError < 0
        '''
        x = -1.5
        soln = "x value needs to be strictly > 0."
        ans = getXError(x)
        self.assertEqual(ans, soln)


    def test_getXError3(self):
        '''getXError actually no error
        '''
        x = 1.7
        soln = "unknown error with x."
        ans = getXError(x)
        self.assertEqual(ans, soln)


    def test_getCoeffError1(self):
        '''getCoeffError list one = 0
        '''
        coeff = [0.5, 0, 1.2]
        soln = "all coefficients need to be strictly > 0."
        ans = getCoeffError(coeff)
        self.assertEqual(ans, soln)


    def test_getCoeffError2(self):
        '''getCoeffError < 0
        '''
        coeff = -1.5
        soln = "coefficient value needs to be strictly > 0."
        ans = getCoeffError(coeff)
        self.assertEqual(ans, soln)


    def test_getCoeffError3(self):
        '''getCoeffError actually no error
        '''
        coeff = 1.7
        soln = "unknown error with coefficients."
        ans = getCoeffError(coeff)
        self.assertEqual(ans, soln)



class HbeTests(unittest.TestCase):

    def test_hbe1(self):
        '''hbe with x float, coeff list
        '''
        x = 10.203
        coeff = [1.5, 1.5, 0.5, 0.5]
        ans = hbe(coeff, x)
        soln = 0.949
        self.assertAlmostEqual(ans, soln, places=3, msg=None, delta=None)

    def test_hbe2(self):
        '''hbe with x float, coeff list, specifying arguments
        '''
        x = 10.203
        coeff = [1.5, 1.5, 0.5, 0.5]
        ans = hbe(x=x, coeff=coeff)
        soln = 0.949
        self.assertAlmostEqual(ans, soln, places=3, msg=None, delta=None)

    def test_hbe3(self):
        '''hbe with x float, coeff list, specifying arguments
        '''
        x = 0.627
        coeff = [1.5, 1.5, 0.5, 0.5]
        ans = hbe(x=x, coeff=coeff)
        soln = 0.0285
        self.assertAlmostEqual(ans, soln, places=3, msg=None, delta=None)


    def test_hbe4(self):
        '''hbe with x list, coeff list, specifying arguments
        '''
        x = [0.627, 10.203]
        coeff = [1.5, 1.5, 0.5, 0.5]
        ans = hbe(coeff=coeff, x=x)
        soln = [0.0285, 0.949]
        # check it is a list
        self.assertTrue(isinstance(ans, list))
        # check lists are equal length and almost equal values
        self.assertEqual(len(ans), len(soln))
        for i in range(len(ans)):
            self.assertAlmostEqual(ans[i], soln[i], places=2, msg=None, delta=None)


    def test_hbe5(self):
        '''hbe with x float, coeff numpy array
        '''
        x = 10.203
        coeff = np.array([1.5, 1.5, 0.5, 0.5])
        ans = hbe(coeff, x)
        soln = 0.949
        self.assertAlmostEqual(ans, soln, places=3, msg=None, delta=None)


    def test_hbe6(self):
        '''hbe with x numpy array, coeff numpy array
        '''
        x = np.array([0.627, 10.203])
        coeff = np.array([1.5, 1.5, 0.5, 0.5])
        ans = hbe(coeff, x)
        soln = np.array([0.0285, 0.949])
        self.assertEqual(len(ans), len(soln))
        for i in range(len(ans)):
            self.assertAlmostEqual(ans[i], soln[i], places=3, msg=None, delta=None)



    def test_hbe7(self):
        '''hbe with x numpy array one element, coeff numpy array
        '''
        x = np.array([0.627])
        coeff = np.array([1.5, 1.5, 0.5, 0.5])
        ans = hbe(coeff, x)
        soln = np.array([0.0285])
        self.assertEqual(len(ans), len(soln))
        for i in range(len(ans)):
            self.assertAlmostEqual(ans[i], soln[i], places=3, msg=None, delta=None)


    def test_hbe8(self):
        '''hbe with x numpy array, coeff list
        '''
        x = np.array([0.627, 10.203])
        coeff = [1.5, 1.5, 0.5, 0.5]
        ans = hbe(coeff, x)
        soln = np.array([0.0285, 0.949])
        self.assertEqual(len(ans), len(soln))
        for i in range(len(ans)):
            self.assertAlmostEqual(ans[i], soln[i], places=3, msg=None, delta=None)


    def test_hbe9(self):
        '''hbe with x list, coeff numpy array
        '''
        x = np.array([0.627, 10.203])
        coeff = [1.5, 1.5, 0.5, 0.5]
        ans = hbe(coeff, x)
        soln = np.array([0.0285, 0.949])
        self.assertEqual(len(ans), len(soln))
        for i in range(len(ans)):
            self.assertAlmostEqual(ans[i], soln[i], places=3, msg=None, delta=None)


class SwTests(unittest.TestCase):

    def test_sw1(self):
        '''sw with x float, coeff list
        '''
        x = 10.203
        coeff = [1.5, 1.5, 0.5, 0.5]
        ans = sw(coeff, x)
        soln = 0.950
        self.assertAlmostEqual(ans, soln, places=3, msg=None, delta=None)


    def test_sw2(self):
        '''sw with x float, coeff list, specifying arguments
        '''
        x = 0.627
        coeff = [1.5, 1.5, 0.5, 0.5]
        ans = sw(x=x, coeff=coeff)
        soln = 0.0657
        self.assertAlmostEqual(ans, soln, places=3, msg=None, delta=None)


    def test_sw3(self):
        '''sw with x list, coeff list, specifying arguments
        '''
        x = [0.627, 10.203]
        coeff = [1.5, 1.5, 0.5, 0.5]
        ans = sw(coeff=coeff, x=x)
        soln = [0.0657, 0.950]
        # check it is a list
        self.assertTrue(isinstance(ans, list))
        # check lists are equal length and almost equal values
        self.assertEqual(len(ans), len(soln))
        for i in range(len(ans)):
            self.assertAlmostEqual(ans[i], soln[i], places=2, msg=None, delta=None)


    def test_sw4(self):
        '''sw with x numpy array, coeff numpy array
        '''
        x = np.array([0.627, 10.203])
        coeff = np.array([1.5, 1.5, 0.5, 0.5])
        ans = sw(coeff, x)
        soln = np.array([0.0657, 0.950])
        self.assertEqual(len(ans), len(soln))
        for i in range(len(ans)):
            self.assertAlmostEqual(ans[i], soln[i], places=3, msg=None, delta=None)


class WfTests(unittest.TestCase):

    def test_wf1(self):
        '''wf with x float, coeff list
        '''
        x = 10.203
        coeff = [1.5, 1.5, 0.5, 0.5]
        ans = wf(coeff, x)
        soln = 0.95105
        self.assertAlmostEqual(ans, soln, places=3, msg=None, delta=None)


    def test_wf2(self):
        '''wf with x float, coeff list, specifying arguments
        '''
        x = 0.627
        coeff = [1.5, 1.5, 0.5, 0.5]
        ans = wf(x=x, coeff=coeff)
        soln = 0.05574
        self.assertAlmostEqual(ans, soln, places=3, msg=None, delta=None)


    def test_wf3(self):
        '''wf with x numpy array, coeff numpy array
        '''
        x = np.array([0.627, 10.203])
        coeff = np.array([1.5, 1.5, 0.5, 0.5])
        ans = wf(coeff, x)
        soln = np.array([0.05574, 0.95105])
        self.assertEqual(len(ans), len(soln))
        for i in range(len(ans)):
            self.assertAlmostEqual(ans[i], soln[i], places=3, msg=None, delta=None)


    def test_wf4(self):
        '''wf with pathological case, calling hbe
        '''
        x = 1
        coeff = [0.9]
        with self.assertWarns(Warning):
            ans = wf(x=x, coeff=coeff)
        soln = hbe(coeff, x)
        self.assertAlmostEqual(ans, soln, places=3, msg=None, delta=None)



class LPBTests(unittest.TestCase):

    def test_lpb_sumofpowers_1(self):
        '''sum_of_powers 1
        '''
        v = np.array([1, 2])
        index = 2
        ans = sum_of_powers(index, v)
        soln = 5
        self.assertEqual(ans, soln)


    def test_lpb_sumofpowers_2(self):
        '''sum_of_powers 2
        '''
        v = np.array([1, 2])
        index = 3
        ans = sum_of_powers(index, v)
        soln = 9
        self.assertEqual(ans, soln)


    def test_lpb_getcumulantvecvectorised_1(self):
        '''get_cumulant_vec_vectorised
        '''
        p = 3
        coeff = np.array([1, 5, 11])
        ans = get_cumulant_vec_vectorised(coeff, p)
        soln = np.array([17, 294, 11656, 732816, 63043968, 6862798080])
        self.assertEqual(len(ans), len(soln))
        for i in range(len(ans)):
            self.assertEqual(ans[i], soln[i])

    def test_lpb_getcumulantvecvectorised_2(self):
        '''get_cumulant_vec_vectorised
        '''
        p = 3
        coeff = np.array([2, 7, 19])
        ans = get_cumulant_vec_vectorised(coeff, p)
        soln = np.array([28, 828, 57680, 6371424, 957288192, 181108200960])
        self.assertEqual(len(ans), len(soln))
        for i in range(len(ans)):
            self.assertEqual(ans[i], soln[i])





    def test_lpb_update_moments(self):
        '''update_moments
        '''
        # made up values
        moment_vec = np.array([1, 4, 18])
        cumul_vec = np.array([2, 11, 20])

        n = 2
        ans = update_moments(n, moment_vec, cumul_vec)
        soln = 2
        self.assertEqual(ans, soln)

        n = 3
        ans = update_moments(n, moment_vec, cumul_vec)
        soln = 30
        self.assertEqual(ans, soln)


    def test_lpb_getmomentsfromcumulants(self):
        '''get_moments_from_cumulants 
        '''
        cumul_vec = np.array([2, 11, 20])
        ans = get_moments_from_cumulants(cumul_vec)
        soln = np.array([2, 15, 94])
        self.assertEqual(len(ans), len(soln))
        for i in range(len(ans)):
            self.assertEqual(ans[i], soln[i])


    def test_lpb_getweightedsumofchisquaredmoments(self):
        '''get_weighted_sum_of_chi_squared_moments
        '''
        coeff = np.array([2, 7, 19])
        p = 3
        ans = get_weighted_sum_of_chi_squared_moments(coeff, p)
        soln = np.array([28, 1612, 149184, 19397904, 3266007360, 675640762560])       
        self.assertEqual( len(ans), len(soln) )
        for i in range(len(ans)):
            self.assertEqual(ans[i], soln[i])


    def test_lpb_getlambdatilde1(self):
        '''get_lambdatilde_1, two pairs of values
        '''
        m1 = 0.5
        m2 = 1.7
        ans = get_lambdatilde_1(m1, m2)
        soln = 5.8
        self.assertEqual(ans, soln)

        m1 = 3.45
        m2 = 7.89
        ans = get_lambdatilde_1(m1, m2)
        soln = -0.3371141
        self.assertAlmostEqual(ans, soln, places=6, msg=None, delta=None)


    def test_lpb_deltaNmatapplied(self):
        '''deltaNmat_applied, 
        '''
        x = 1.23
        m_vec = np.array([1, 5, 14, 27, 59, 123])
        N = 3
        ans = deltaNmat_applied(x, m_vec, N)
        #         [,1]      [,2]      [,3]       [,4]
        #[1,] 1.000000 1.0000000 2.2421525 1.81445864
        #[2,] 1.000000 2.2421525 1.8144586 0.74612220
        #[3,] 2.242152 1.8144586 0.7461222 0.27540797
        #[4,] 1.814459 0.7461222 0.2754080 0.08030148
        soln = np.array([[1.000000, 1.0000000, 2.2421525, 1.81445864],
            [1.000000, 2.2421525, 1.8144586, 0.74612220],
            [2.242152, 1.8144586, 0.7461222, 0.27540797],
            [1.814459, 0.7461222, 0.2754080, 0.08030148]])

        self.assertEqual(ans.shape, soln.shape)
        for i in range(ans.shape[0]):
            for j in range(ans.shape[1]):
                self.assertAlmostEqual(ans[i,j], soln[i,j], places=4, msg=None, delta=None)


    def test_lpb_deltaNmat(self):
        '''determinant of deltaNmat
        '''
        x = 1.23
        m_vec = np.array([1, 5, 14, 27, 59, 123])
        N = 3
        ans = det_deltaNmat(x, m_vec, N)
        soln = 1.218685
        self.assertAlmostEqual(ans, soln, places=4, msg=None, delta=None)


    def test_lpb_getlambdatildep(self):
        '''get_lambdatilde_p
        '''
        x = 1.23
        p = 3
        coeff = np.array([0.5, 1.2, 2.3])
        m_vec = get_weighted_sum_of_chi_squared_moments(coeff, p)
        lambdatilde_1 = get_lambdatilde_1(m_vec[0], m_vec[1])
        bisect_tol=1e-9
        ans = get_lambdatilde_p(lambdatilde_1, p, m_vec, bisect_tol)

        soln = 0.681807
        self.assertAlmostEqual(ans, soln, places=4, msg=None, delta=None)

    def test_lpb_getbasevector(self):
        '''get_base_vector two examples
        '''
        n = 4
        i = 0
        ans = get_base_vector(n, i)
        soln = np.array([1, 0, 0, 0])

        self.assertEqual( len(ans), len(soln) )
        for i in range(len(ans)):
            self.assertEqual(ans[i], soln[i])

        n = 4
        i = 2
        ans = get_base_vector(n, i)
        soln = np.array([0, 0, 1, 0])

        self.assertEqual( len(ans), len(soln) )
        for i in range(len(ans)):
            self.assertEqual(ans[i], soln[i])


    def test_lpb_getithcoeffofStildepoly(self):
        '''get_ith_coeff_of_Stilde_poly
        '''
        mat = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]) * 0.1
        i = 1
        ans = get_ith_coeff_of_Stilde_poly(i, mat)
        soln = 0.06
        self.assertAlmostEqual(ans, soln, places=7, msg=None, delta=None)
 

    def test_lpb_getStildepolycoeff(self):
        '''get_Stilde_poly_coeff
        '''
        mat = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]) * 0.1
        ans = get_Stilde_poly_coeff(mat)
        soln = np.array([-0.03, 0.06, -0.03])

        self.assertEqual( len(ans), len(soln) )
        for i in range(len(ans)):
            self.assertAlmostEqual(ans[i], soln[i], places=7, msg=None, delta=None)


    def test_lpb_getVDMbvec(self):
        '''get_VDM_b_vec
        '''
        mat = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]) * 0.1
        ans = get_VDM_b_vec(mat)
        soln = np.array([0.1, 0.4])

        self.assertEqual( len(ans), len(soln) )
        for i in range(len(ans)):
            self.assertAlmostEqual(ans[i], soln[i], places=7, msg=None, delta=None)


    def test_lpb_getvandermonde(self):
        '''get_vandermonde
        '''
        vec = np.array([0.1, 1.2, 3.5])
        ans = get_vandermonde(vec)
        #      [,1] [,2]  [,3]
        # [1,] 1.00 1.00  1.00
        # [2,] 0.10 1.20  3.50
        # [3,] 0.01 1.44 12.25
        soln = np.array([[1.00, 1.00,  1.00], 
                         [0.10, 1.20,  3.50], 
                         [0.01, 1.44, 12.25]] )
        self.assertEqual(ans.shape, soln.shape)
        for i in range(ans.shape[0]):
            for j in range(ans.shape[1]):
                self.assertAlmostEqual(ans[i,j], soln[i,j], places=4, msg=None, delta=None)


    def test_lpb_getmupolyvec(self):
        '''run steps until mu_poly_coeff_vec is generated
        '''
        coeff = np.array([0.5, 1.2, 3.4, 5.6])
        p = 3
        bisect_tol = 1e-9
        m_vec = get_weighted_sum_of_chi_squared_moments(coeff, p)
        lambdatilde_1 = get_lambdatilde_1(m_vec[0], m_vec[1])
        lambdatilde_p = get_lambdatilde_p(lambdatilde_1, p, m_vec, bisect_tol)
        M_p = deltaNmat_applied(lambdatilde_p, m_vec, p)

        ans = get_Stilde_poly_coeff(M_p)
        soln = np.array([-5698007.906, 1725987.319, -156331.434, 4342.343])
        self.assertEqual( len(ans), len(soln) )
        for i in range(len(ans)):
            self.assertAlmostEqual(ans[i], soln[i], places=2, msg=None, delta=None)


    def test_lpb_getrealpolyroots(self):
        '''get_real_poly_roots
        '''
        coeff = np.array([0.5, 1.2, 3.4, 5.6])
        p = 3
        bisect_tol = 1e-9
        m_vec = get_weighted_sum_of_chi_squared_moments(coeff, p)
        lambdatilde_1 = get_lambdatilde_1(m_vec[0], m_vec[1])
        lambdatilde_p = get_lambdatilde_p(lambdatilde_1, p, m_vec, bisect_tol)
        M_p = deltaNmat_applied(lambdatilde_p, m_vec, p)
        mu_poly_coeff_vec = get_Stilde_poly_coeff(M_p)
        ans = get_real_poly_roots(mu_poly_coeff_vec)
        soln = np.array([6.103128, 12.037864, 17.860640])
        self.assertEqual( len(ans), len(soln) )
        for i in range(len(ans)):
            self.assertAlmostEqual(ans[i], soln[i], places=2, msg=None, delta=None)


    def test_lpb_genandsolveVDMsystem(self):
        '''gen_and_solve_VDM_system
        '''
        coeff = np.array([0.5, 1.2, 3.4, 5.6])
        p = 3
        bisect_tol = 1e-9
        m_vec = get_weighted_sum_of_chi_squared_moments(coeff, p)
        lambdatilde_1 = get_lambdatilde_1(m_vec[0], m_vec[1])
        lambdatilde_p = get_lambdatilde_p(lambdatilde_1, p, m_vec, bisect_tol)
        M_p = deltaNmat_applied(lambdatilde_p, m_vec, p)
        mu_poly_coeff_vec = get_Stilde_poly_coeff(M_p)
        mu_roots = get_real_poly_roots(mu_poly_coeff_vec)
        ans = gen_and_solve_VDM_system(M_p, mu_roots)
        soln = np.array([0.3699345, 0.4827818,  0.1472837])
        self.assertEqual( len(ans), len(soln) )
        for i in range(len(ans)):
            self.assertAlmostEqual(ans[i], soln[i], places=5, msg=None, delta=None)



    def test_lpb_lambdatildep(self):
        '''get_lambdatilde_p
        '''
        coeff = np.array([0.5, 1.2, 3.4, 5.6])
        p = 3
        bisect_tol = 1e-9
        m_vec = get_weighted_sum_of_chi_squared_moments(coeff, p)
        lambdatilde_1 = get_lambdatilde_1(m_vec[0], m_vec[1])
        ans = get_lambdatilde_p(lambdatilde_1, p, m_vec, bisect_tol)
        soln = 0.5583305
        self.assertAlmostEqual(ans, soln, places=7, msg=None, delta=None)


    def test_lpb_getmixedpvalvec(self):
        '''get_mixed_pval_vec
        '''
        pi_vec = np.array([0.3699345, 0.4827818,  0.1472837])
        mu_roots = np.array([6.103128, 12.037864, 17.860640])
        lambdatilde_p = 0.5583305
        q_vec = np.array([0.627, 10.203])
        ans = get_mixed_pval_vec(q_vec, mu_roots, pi_vec, lambdatilde_p)
        soln = np.array([0.01404122, 0.60892878])
        self.assertEqual( len(ans), len(soln) )
        for i in range(len(ans)):
            self.assertAlmostEqual(ans[i], soln[i], places=5, msg=None, delta=None)


    def test_lpb_getmixedpvalvec_2(self):
        '''get_mixed_pval_vec for float
        '''
        pi_vec = np.array([0.3699345, 0.4827818,  0.1472837])
        mu_roots = np.array([6.103128, 12.037864, 17.860640])
        lambdatilde_p = 0.5583305
        q_vec = 0.627
        ans = get_mixed_pval_vec(q_vec, mu_roots, pi_vec, lambdatilde_p)
        soln = 0.01404122
        self.assertAlmostEqual(ans, soln, places=7, msg=None, delta=None)


    def test_lpb4_1(self):
        '''test lpb4
        '''
        ans = lpb4([1.5, 1.5, 0.5, 0.5], 10.203)
        soln = 0.9500092
        self.assertAlmostEqual(ans, soln, places=6, msg=None, delta=None)


    def test_lpb4_2(self):
        '''test lpb4 with warning
        '''
        # in this case, will call hbe
        x = 2.708
        coeff = [0.5, 0.3, 0.2]
        soln = hbe(coeff, x)
        with self.assertWarns(Warning) as context:
            ans = lpb4(coeff, x)
        self.assertAlmostEqual(ans, soln, places=6, msg=None, delta=None)


    def test_lpb4_3(self):
        '''test lpb4 with warning
        '''
        # in this case, will call hbe
        x = 2.708
        coeff = [0.5, 0.3]
        soln = hbe(coeff, x)
        with self.assertWarns(Warning) as context:
            ans = lpb4(coeff, x, p=3)
        self.assertAlmostEqual(ans, soln, places=6, msg=None, delta=None)




class InputErrorTests(unittest.TestCase):

    def test_hbeInputError1(self):
        '''test hbe input error coeff
        '''
        x = 10.203
        coeff = [-1.5, 1.5, 0.5, 0.5]
        with self.assertRaises(Exception) as context:
            hbe(coeff, x)
        self.assertTrue(getCoeffError(coeff) in str(context.exception))

    def test_hbeInputError2(self):
        '''test hbe input error x
        '''
        x = [-1.2, 9]
        coeff = [1.5, 1.5, 0.5, 0.5]
        with self.assertRaises(Exception) as context:
            hbe(coeff, x)
        self.assertTrue(getXError(x) in str(context.exception))

    def test_swInputError1(self):
        '''test sw input error coeff
        '''
        x = 10.203
        coeff = [-1.5, 1.5, 0.5, 0.5]
        with self.assertRaises(Exception) as context:
            sw(coeff, x)
        self.assertTrue(getCoeffError(coeff) in str(context.exception))

    def test_swInputError2(self):
        '''test sw input error x
        '''
        x = [-1.2, 9]
        coeff = [1.5, 1.5, 0.5, 0.5]
        with self.assertRaises(Exception) as context:
            sw(coeff, x)
        self.assertTrue(getXError(x) in str(context.exception))

    def test_wfInputError1(self):
        '''test wf input error coeff
        '''
        x = 10.203
        coeff = [-1.5, 1.5, 0.5, 0.5]
        with self.assertRaises(Exception) as context:
            wf(coeff, x)
        self.assertTrue(getCoeffError(coeff) in str(context.exception))

    def test_wfInputError2(self):
        '''test wf input error x
        '''
        x = [-1.2, 9]
        coeff = [1.5, 1.5, 0.5, 0.5]
        with self.assertRaises(Exception) as context:
            wf(coeff, x)
        self.assertTrue(getXError(x) in str(context.exception))

    def test_lpb4InputError1(self):
        '''test lpb4 input error coeff
        '''
        x = 10.203
        coeff = [-1.5, 1.5, 0.5, 0.5]
        with self.assertRaises(Exception) as context:
            lpb4(coeff, x)
        self.assertTrue(getCoeffError(coeff) in str(context.exception))

    def test_lpb4InputError2(self):
        '''test lpb4 input error x
        '''
        x = [-1.2, 9]
        coeff = [1.5, 1.5, 0.5, 0.5]
        with self.assertRaises(Exception) as context:
            lpb4(coeff, x)
        self.assertTrue(getXError(x) in str(context.exception))



