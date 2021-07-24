"""
In this assignment you should find the intersection points for two functions.
"""

import numpy as np
import time
import random


class Assignment2:
    def __init__(self):
        """
        Here goes any one time calculation that need to be made before 
        solving the assignment for specific functions. 
        """

        pass
    def myBisection(self, a: float, b: float, f: callable, err: float, epsilon=0.01) -> list:

        if (b - a < epsilon):
            toReturn=[]
            if(toReturn is None):
                print("none")
            return toReturn

        newA = a
        newB = b
        fA=f(newA)
        fB=f(newB)

        # find a and b with opposite signs by random
        #---------------------------------------
        countTries=0
        while (fA * fB >= 0 and countTries<100):
            newA = np.random.uniform(a, b)
            newB = np.random.uniform(a, b)
            fA = f(newA)
            fB = f(newB)
            if (newA > newB):
                tmp = newB
                newB = newA
                newA = tmp
            countTries=countTries+1
        #-----------------------------------

        #find c s.t f(c)=0 with binary search
        fc=0

        if(countTries>=10 and fA*fB>=0):
            toReturn=[]
            if (toReturn is None):
                print("none")
            return toReturn

        c = newA
        stillRun = 1
        while (stillRun):
            c = (newA + newB) / 2
            fc=f(c)
            if (abs(fc) <= err):
                stillRun = 0
            elif (fc * f(newA) < 0):
                newB = c
            else:
                newA = c



        positiveSlope=0
        aboveXaxis=0
        #checkOneStep
        if(f(c+epsilon)>fc):
            positiveSlope=1
        else:
            positiveSlope=0

        #check where on Y axis
        if(fc>0):
            aboveXaxis=1
        else:
            aboveXaxis=0

        #adjust the c for left and right
        cForLeft=c
        cForRight=c

        FCforLeft=fc
        FCforRight=fc

        while (abs(FCforLeft) <= err or abs(FCforRight)<=err):
            if(abs(FCforLeft) <= err):
                cForLeft=cForLeft-epsilon
                FCforLeft=f(cForLeft)
            if(abs(FCforRight)<=err):
                cForRight=cForRight+epsilon
                FCforRight=f(cForRight)



        left:list = self.myBisection(a, cForLeft, f, err)

        right:list = self.myBisection(cForRight, b,f , err)
        if (left is None):
            print("none")

        if  (right is None):
            print("none")

        left.extend(right)
        left.append(c)

        toReturn=left
        if (toReturn is None):
            print ("none")
        return toReturn

    def intersections(self, f1: callable, f2: callable, a: float, b: float, maxerr=0.001) -> callable:
        """
        Find as many intersection points as you can. The assignment will be
        tested on functions that have at least two intersection points, one
        with a positive x and one with a negative x.


        Parameters
        ----------
        f1 : callable
            the first given function
        f2 : callable
            the second given function
        a : float
            beginning of the interpolation range.
        b : float
            end of the interpolation range.
        maxerr : float
            An upper bound on the difference between the
            function values at the approximate intersection points.


        Returns
        -------
        X : iterable of approximate intersection Xs such that for each x in X:
            |f1(x)-f2(x)|<=maxerr.

        """

        # replace this line with your solution

        def sub(x: float):
            return (f1(x) - f2(x))


        X = self.myBisection(a, b, sub, maxerr)
        return X





##########################################################################


import unittest
from sampleFunctions import *
from tqdm import tqdm
import matplotlib.pyplot as plt


class TestAssignment2(unittest.TestCase):

    def test_dor(self):
        #               9   8   7   6   5   4   3    2    1   0
        #f1 = np.poly1d([0,  0,  0,  1,  0,  3,  -9,  0,   5,  3])
        #f2 = np.poly1d([1,  0,  0,  0,  0,  3,  -12,  0,  0,  5])

        #f1 = np.poly1d([1, 0, 0, 0])

        def f1(x):
            return np.sin(20*x)
        f2 = np.poly1d([1 , 0])

        xp1 = np.arange(-20, 20, 0.1)
        yp1 = f1(xp1)

        xp2 = np.arange(-20, 20, 0.1)
        yp2 = f2(xp2)

        fig, (ax11) = plt.subplots(nrows=1, ncols=1, sharex=False, figsize=(12, 4))
        ax11.plot(xp1, yp1, color="red")
        #ax11.plot(xp2, yp2, color="blue")
        plt.show()

        ass2=Assignment2()
        X = ass2.intersections(f1, f2, -2, 2, maxerr=0.001)
        print(X)




"""

    def test_sqr(self):

        ass2 = Assignment2()
        print("hellp")
        f1 = np.poly1d([-1, 0, 1])
        f2 = np.poly1d([1, 0, -1])
        fsin = np.sin
        fcos = np.cos

        xp1 = np.arange(-20, 20, 0.1)
        yp1 = f1(xp1)

        xp2 = np.arange(-20, 20, 0.1)
        yp2 = f2(xp2)

        def sub(x):
            return f1(x)-f2(x)

        ypsub=sub(xp1)



        fig, (ax11) = plt.subplots(nrows=1, ncols=1, sharex=False, figsize=(12, 4))
        ax11.plot(xp1, yp1, color="red")
        ax11.plot(xp2, yp2, color="green")
        ax11.plot(xp1, ypsub, color="orange")


        plt.show()

        xp_sin = np.arange(-300, 300, 10)
        yp_sin = fsin(10 * xp_sin)

        xp_cos = np.arange(-300, 300, 10)
        yp_cos = fcos(xp_cos)

        t = np.sin(100)
        fig, (ax12) = plt.subplots(nrows=1, ncols=1, sharex=False, figsize=(12, 4))



        xp_sub = np.arange(-300, 300, 10)
        yp_sub = list(map(sub, xp_sub))

        ax12.plot(xp_sin, yp_sin, color="red")
        ax12.plot(xp_cos, yp_cos, color="green")
        ax12.plot(xp_sub, yp_sub, color="blue")
        #plt.show()
        X = ass2.intersections(f1, f2, -1, 1, maxerr=0.001)

        for x in X:
            self.assertGreaterEqual(0.001, f1(x) - f2(x))


    def test_poly(self):

        ass2 = Assignment2()

        f1, f2 = randomIntersectingPolynomials(10)

        X = ass2.intersections(f1, f2, -1, 1, maxerr=0.001)

        for x in X:
            self.assertGreaterEqual(0.001, abs(f1(x) - f2(x)))
"""

if __name__ == "__main__":
    unittest.main()
