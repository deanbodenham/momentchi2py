import unittest
import numpy as np
from momentchi2.methods import addOne
from momentchi2.methods import hbe
from momentchi2.methods import sw
from momentchi2.methods import wf



class BasicTests(unittest.TestCase):

    def test_basic1(self):
        '''Testing if two numbers are equal, just to get started
        '''
        x = 0
        y = 0
        self.assertEqual(x, y)


class BasicMomentchiTests(unittest.TestCase):

    def test_addone1(self):
        '''Adding one
        '''
        ans = addOne(2)
        soln = 3
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



