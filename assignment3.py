"""
In this assignment you should find the area enclosed between the two given functions.
The rightmost and the leftmost x values for the integration are the rightmost and 
the leftmost intersection points of the two functions. 

The functions for the numeric answers are specified in MOODLE. 


This assignment is more complicated than Assignment1 and Assignment2 because: 
    1. You should work with float32 precision only (in all calculations) and minimize the floating point errors. 
    2. You have the freedom to choose how to calculate the area between the two functions. 
    3. The functions may intersect multiple times. Here is an example: 
        https://www.wolframalpha.com/input/?i=area+between+the+curves+y%3D1-2x%5E2%2Bx%5E3+and+y%3Dx
    4. Some of the functions are hard to integrate accurately. 
       You should explain why in one of the theoretical questions in MOODLE. 

"""

import numpy as np
import time
import random

from assignment2 import Assignment2


class Assignment3:
    def __init__(self):
        """
        Here goes any one time calculation that need to be made before 
        solving the assignment for specific functions. 
        """

        pass

    def integrate(self, f: callable, a: float, b: float, n: int) -> np.float32:
        """
        Integrate the function f in the closed range [a,b] using at most n 
        points. Your main objective is minimizing the integration error. 
        Your secondary objective is minimizing the running time. The assignment
        will be tested on variety of different functions. 
        
        Integration error will be measured compared to the actual value of the 
        definite integral. 
        
        Note: It is forbidden to call f more than n times. 
        
        Parameters
        ----------
        f : callable. it is the given function
        a : float
            beginning of the integration range.
        b : float
            end of the integration range.
        n : int
            maximal number of points to use.

        Returns
        -------
        np.float32
            The definite integral of f between a and b
        """

        # replace this line with your solution
        b32=np.float32(b)
        a32=np.float32(a)
        n32=np.float32(n)

        #simpson(a, b, n, f):
        sum = np.float32(0.0)
        inc = (b32 - a32) / n32
        if (n32%2!=0.0):
            n32=n32-1.0
        for k in range(n + 1):
            x = a32 + (k * inc)
            summand = f(x)
            if(summand==np.float32("inf") or summand==np.float32("-inf")):
                summand=0.0

            if (k != 0.0) and (k != n):
                summand *= (2.0 + (2.0 * (k % 2.0)))
            print("x: "+str(x)+", summand: "+str(summand))
            sum += summand
        result= ((b32 - a32) / (3.0 * n32)) * sum

        return np.float32(result)

    def areabetween(self, f1: callable, f2: callable) -> np.float32:
        """
        Finds the area enclosed between two functions. This method finds 
        all intersection points between the two functions to work correctly. 
        
        Example: https://www.wolframalpha.com/input/?i=area+between+the+curves+y%3D1-2x%5E2%2Bx%5E3+and+y%3Dx

        Note, there is no such thing as negative area. 
        
        In order to find the enclosed area the given functions must intersect 
        in at least two points. If the functions do not intersect or intersect 
        in less than two points this function returns NaN.  
        

        Parameters
        ----------
        f1,f2 : callable. These are the given functions

        Returns
        -------
        np.float32
            The area between function and the X axis

        """

        # replace this line with your solution


        ass2=Assignment2()

        X=ass2.intersections(f1,f2,-100,100)
        X.sort()
        #j=i+1
        Stotal=0
        for i in range(len(X)-1):
            xi=X[i]
            xj=X[i+1]
            Sf1=self.integrate(f1,xi,xj,100)
            Sf2=self.integrate(f2,xi,xj,100)
            Sf1f2= abs(Sf1-Sf2)
            Stotal=Stotal+Sf1f2






        result = Stotal

        return result


##########################################################################


import unittest
from sampleFunctions import *
from tqdm import tqdm
import matplotlib.pyplot as plt


class TestAssignment3(unittest.TestCase):
    def test_areaBetween(self):
        ass3=Assignment3()

        f1=np.poly1d([-10,0,5])
        f2=np.poly1d([-1,0,2])
        ass2=Assignment2()
        X=ass2.intersections(f1,f2,-100,100)
        area=ass3.areabetween(f1,f2)
        print(area)


"""
    def test_dor(self):

        ass3=Assignment3()
        f1=np.poly1d([1, 0])
        print("#### first method ######")
        r=ass3.integrate(f1,0,1,10)
        print(r)

    def test_integrate_float32(self):
        ass3 = Assignment3()
        f1 = np.poly1d([-1, 0, 1])
        print("#### second method ######")
        r = ass3.integrate(f1, -1, 1, 10)

        self.assertEquals(r.dtype, np.float32)

    def test_integrate_hard_case(self):
        ass3 = Assignment3()
        f1 = strong_oscilations()
        xp1 = np.arange(-300, 300, 10)
        yp1 = f1(xp1)
        fig, (ax12) = plt.subplots(nrows=1, ncols=1, sharex=False, figsize=(12, 4))
        ax12.plot(xp1, yp1, color="red")
        plt.show()
        print("#### third method ######")
        r = ass3.integrate(f1, 0.09, 10, 20)
        true_result = -7.78662 * 10 ** 33
        self.assertGreaterEqual(0.001, (r - true_result) / true_result)
"""


if __name__ == "__main__":
    unittest.main()
