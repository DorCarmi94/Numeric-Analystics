"""
In this assignment you should interpolate the given function.
"""

import numpy as np
import time
import random



class Assignment1:
    def __init__(self):
        """
        Here goes any one time calculation that need to be made before 
        starting to interpolate arbitrary functions.
        """

        self._ranges=[]


    def interpolate(self, f: callable, a: float, b: float, n: int) -> callable:
        """
        Interpolate the function f in the closed range [a,b] using at most n 
        points. Your main objective is minimizing the interpolation error.
        Your secondary objective is minimizing the running time. 
        The assignment will be tested on variety of different functions with 
        large n values. 
        
        Interpolation error will be measured as the average absolute error at 
        2*n random points between a and b. See test_with_poly() below. 

        Note: It is forbidden to call f more than n times. 

        Note: This assignment can be solved trivially with running time O(n^2)
        or it can be solved with running time of O(n) with some preprocessing.
        **Accurate O(n) solutions will receive higher grades.** 
        
        Note: sometimes you can get very accurate solutions with only few points, 
        significantly less than n. 
        
        Parameters
        ----------
        f : callable. it is the given function
        a : float
            beginning of the interpolation range.
        b : float
            end of the interpolation range.
        n : int
            maximal number of points to use.

        Returns
        -------
        The interpolating function.
        """

        #create the array of points
        pointsArr=[]
        deltaOfXBetweenPoints=(float)(b-a)/float(n)
        xCounter=a

        xs=[]
        ys=[]
        for i in range(n):
            if (xCounter>b):
                xCounter=b
            yy=float(f(xCounter))
            xx=float(xCounter)


            pointsArr.append((xx,yy))
            xs.append(xx)
            ys.append(yy)
            xCounter=xCounter+deltaOfXBetweenPoints

        #lagrange func

        polyAcc = np.poly1d([0])
        for i in range(n):
            Li=np.poly1d([1])
            xi,yi=pointsArr[i]
            for j in range(n):
                if i==j:
                    continue
                xj,yj=pointsArr[j]

                m=1/(xi-xj)
                k=(-1)*(xj)/(xi-xj)
                nextPolyArg = np.poly1d([m, k])
                Li = Li * nextPolyArg
            polyAcc=polyAcc+(Li*yi)
        return polyAcc


        """
        def Lagrange(x):
            polyAccumulate=0
            for i in range(n):
                xi,yi=pointsArr[i]
                polyAccumulate=polyAccumulate + yi*lagrangePai(pointsArr,n,i,xi,x)

            return polyAccumulate

        def lagrangePai(points,n,i,xi,x):
                Li=1
                for j in range(n):
                    xj,yj=points[j]
                    if j == i:
                        continue
                    nextPolyArg=x
                    nextPolyArg=nextPolyArg-xj
                    nextPolyArg=nextPolyArg/float((xi-xj))
                    Li=Li*nextPolyArg

                return Li
        return lagrange(xs,ys)
        """


    def interpolate2(self, f: callable, a: float, b: float, n: int) -> callable:
        """
        Interpolate the function f in the closed range [a,b] using at most n
        points. Your main objective is minimizing the interpolation error.
        Your secondary objective is minimizing the running time.
        The assignment will be tested on variety of different functions with
        large n values.

        Interpolation error will be measured as the average absolute error at
        2*n random points between a and b. See test_with_poly() below.

        Note: It is forbidden to call f more than n times.

        Note: This assignment can be solved trivially with running time O(n^2)
        or it can be solved with running time of O(n) with some preprocessing.
        **Accurate O(n) solutions will receive higher grades.**

        Note: sometimes you can get very accurate solutions with only few points,
        significantly less than n.

        Parameters
        ----------
        f : callable. it is the given function
        a : float
            beginning of the interpolation range.
        b : float
            end of the interpolation range.
        n : int
            maximal number of points to use.

        Returns
        -------
        The interpolating function.
        """

        # create the array of points
        pointsArr = []
        deltaOfXBetweenPoints = (float)(b - a) / float(n)
        xCounter = a

        xs = []
        ys = []
        for i in range(n):
            if (xCounter > b):
                xCounter = b
            yy = float(f(xCounter))
            xx = float(xCounter)

            pointsArr.append((xx, yy))
            xs.append(xx)
            ys.append(yy)
            xCounter = xCounter + deltaOfXBetweenPoints

        return lagrange(xs,ys)

    """
    def Lagrange(x):
        polyAccumulate=0
        for i in range(n):
            xi,yi=pointsArr[i]
            polyAccumulate=polyAccumulate + yi*lagrangePai(pointsArr,n,i,xi,x)

        return polyAccumulate

    def lagrangePai(points,n,i,xi,x):
            Li=1
            for j in range(n):
                xj,yj=points[j]
                if j == i:
                    continue
                nextPolyArg=x
                nextPolyArg=nextPolyArg-xj
                nextPolyArg=nextPolyArg/float((xi-xj))
                Li=Li*nextPolyArg

            return Li
    return lagrange(xs,ys)
    """

    def inteprpolateWithLinerSplines(self, f: callable, a: float, b: float, n: int) -> callable:
        pointsArr = []
        deltaOfXBetweenPoints = (float)(b - a) / float(n)
        xCounter = a

        xs = []
        ys = []
        for i in range(n):
            if (xCounter > b):
                xCounter = b
            yy = float(f(xCounter))
            xx = float(xCounter)
            self._ranges.append((xx,yy))
            xs.append(xx)
            ys.append(yy)
            xCounter = xCounter + deltaOfXBetweenPoints

        def F(x:float):
            y=float("nan")
            for i in range(len(self._ranges)):
                if(i==n-1):
                    break
                xi,yi= self._ranges[i]
                j=i+1
                xj,yj=self._ranges[j]
                if (x >= xi and x<xj):
                    m=(yj-yi)/(xj-xi)
                    y=m*(x-xj)+yj
                    break
                elif(x<xi):
                    break
                else:
                    continue
            return y

        return F

import unittest
from functionUtils import *
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.interpolate import lagrange


class TestAssignment1(unittest.TestCase):
    """

    code for print plot:
    import matplotlib.pyplot as plt
    xp=np.arange(-200,200,5)
    yp=f(xp)
    fig, (ax12)= plt.subplots(nrows=1, ncols=1,sharex=False, figsize=(12,4))
    ax12.plot(xp, yk, color="blue")
    plt.show()



    """


    def dfunc(self,x):
        return np.power(x,3)


    def test_with_poly(self):

        T = time.time()

        ass1 = Assignment1()
        mean_err = 0

        d = 10
        for i in tqdm(range(100)):
            a = np.random.randn(d)

            f = np.poly1d(a)
            #f = self.dfunc

            #ff = ass1.interpolate(f, -10, 10, 300 + 1)
            #fff = ass1.interpolate2(f, -10, 10, 300 + 1)
            ff=ass1.inteprpolateWithLinerSplines(f,-10,10,300+1)
            #yk=ff(xp)
            #ax12.plot(xp,yk,color="blue")
            #plt.show()
            xs = np.random.random(200)
            xp=np.arange(-200,200,5)
            yp=f(xp)
            fig, (ax12)= plt.subplots(nrows=1, ncols=1,sharex=False, figsize=(12,4))
            #ax12.plot(xp,yp,color="orange")
            #yk=ff(xp)
            #ax12.plot(xp, yk, color="blue")

            #plt.show()


            err = 0
            err2=0
            mean_err2=0
            for x in xs:
                yy = ff(x)
                y = f(x)
                err += abs(y - yy)
                #err2+=abs(y-yyy)

            err = err / 200
            err2=err2/200
            #mean_err2+=err2
            mean_err += err
        mean_err = mean_err / 100
        #mean_err2=mean_err2/100

        T = time.time() - T
        print(T)
        print(mean_err)
        #print(mean_err2)


    def test_with_poly_restrict(self):
        ass1 = Assignment1()
        a = np.random.randn(5)
        f = RESTRICT_INVOCATIONS(10)(np.poly1d(a))
        ff = ass1.interpolate(f, -10, 10, 10)
        xs = np.random.random(20)
        for x in xs:
            yy = ff(x)

"""
    def test_dor(self):
        print("dor test:")
        ass1 = Assignment1()

        xp = np.arange(-10, 10, 0.5)
        y_original =self.dfunc(xp)

        fig, (ax12) = plt.subplots(nrows=1, ncols=1, sharex=False, figsize=(12, 4))
        #ax12.plot(xp, y_original, color="blue")

        f = ass1.inteprpolateWithLinerSplines(self.dfunc, -10, 10, 300)
        yp_new = []
        for num in xp:
            yp_new.append(f(num))


        ax12.plot(xp, yp_new, color="orange")

        plt.show()
"""






if __name__ == "__main__":
    unittest.main()
