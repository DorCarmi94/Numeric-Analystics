"""
In this assignment you should fit a model function of your choice to data 
that you sample from a contour of given shape. Then you should calculate
the area of that shape. 

The sampled data is very noisy so you should minimize the mean least squares 
between the model you fit and the data points you sample.  

During the testing of this assignment running time will be constrained. You
receive the maximal running time as an argument for the fitting method. You 
must make sure that the fitting function returns at most 5 seconds after the 
allowed running time elapses. If you know that your iterations may take more 
than 1-2 seconds break out of any optimization loops you have ahead of time.

Note: You are allowed to use any numeric optimization libraries and tools you want
for solving this assignment. 
Note: !!!Despite previous note, using reflection to check for the parameters 
of the sampled function is considered cheating!!! You are only allowed to 
get (x,y) points from the given shape by calling sample(). 
"""

import numpy as np
import time
import random

from NewVersion.assignment4A import Assignment4A
from functionUtils import AbstractShape


class MyShape(AbstractShape):
    # change this class with anything you need to implement the shape
    def __init__(self, f:callable):
        pass

        #init
        states = ["forwardInit","forward", "backward"]
        currentState = states[0]
        self._fSamples=f

        numberofSamples=500
        counter = 0

        self._functions=[]
        self._xStratXendForEachFunc=[]
        newXi, newYi = self._fSamples()

        currSamples=[]
        xs=[]
        ys=[]

        currSamples.append((newXi,newYi))
        xs.append(newXi)
        ys.append(newYi)

        prevXi=newXi
        prevYi=newYi

        firstXi=newXi
        lastXi=newXi
        firstYi=newYi
        lastYi=newYi

        ass4A = Assignment4A()



        #state machine
        tenCount=0
        while (counter<=numberofSamples):
            prevXi, prevYi = newXi, newYi
            lastXi,lastYi = newXi,newYi
            newXi, newYi = self._fSamples()
            currSamples.append((newXi,newYi))
            xs.append(newXi)
            ys.append(newYi)

            if (tenCount==10):
                newLine=self.linear_leastSquares(currSamples,xs,ys)
                currSamples=[]
                xs=[]
                ys=[]
                self._functions.append(newLine)
                tenCount=0
            counter+=1
            tenCount+=1

        tenCount+=1



    def contour(self, n: int):
        numOfFuncs=len(self._functions)
        numOfSamplesForEachFunc=int(n/numOfFuncs)
        samplesToReturn=[]

        for i in range(numOfFuncs):
            currF=self._functions[i]
            currX=currF.getXstart()
            step=abs(currF.getXend()-currF.getXstart())/np.float32(numOfSamplesForEachFunc)
            for j in range(numOfSamplesForEachFunc):
                currY=currF.getY(currX)
                samplesToReturn.append((currX,currY))
                currX=currX+step
        return samplesToReturn

    def linear_leastSquares(self,samples,Xs,Ys)->callable:

            """
            (x1,y1), (x2,y2)....(xn,yn)

            =>


            A(vertical)=    [       x1      1
                                    x2      1
                                    ...
                                    xn      1   ]

            At(horizental)=     [   x1  x2  ..  xn
                                    1   1   ..  1   ]

            b=Ys= [     y1
                        y2
                        ..
                        yn  ]

            At * A = At * b

            """
            leng=len(samples)
            for i in range(leng):
                B = [1] * leng
                mmHorizental_At = np.matrix([Xs, B])
                mmVertical_A = mmHorizental_At.T
                AtA = mmHorizental_At * mmVertical_A
                matrixY=np.matrix(Ys)
                vectorY=matrixY.T
                Atb = mmHorizental_At * vectorY

                slv = np.linalg.solve(AtA, Atb)
                m = slv[0, 0]
                n = slv[1, 0]
                line=Line(m,n,Xs[0],Xs[-1])
                return line

class Line:
    def __init__(self,m,n,startX,endX):
        self._m=m
        self._n=n
        self._startX=startX
        self._endX=endX

    def getM(self):
        return self._m

    def getN(self):
        return self._n

    def getXstart(self):
        return self._startX

    def getXend(self):
        return self._endX

    def getY(self,x):
        return self._m*x+self._n;

    def getFirstY(self,x):
        return self.getY(self._startX)

    def getLastY(self,x):
        return self.getY(self._endX)

class Assignment4:
    def __init__(self):
        """
        Here goes any one time calculation that need to be made before 
        solving the assignment for specific functions. 
        """

        pass

    def area(self,contour: callable, maxerr=0.001)->np.float32:
        """
        Compute the area of the shape with the given contour. 

        Parameters
        ----------
        contour : callable
            Same as AbstractShape.contour 
        maxerr : TYPE, optional
            The target error of the area computation. The default is 0.001.

        Returns
        -------
        The area of the shape.

        """
        maxFP=np.float32(maxerr)
        n=np.int32((np.float32(1.0))/(maxFP*np.float32(10)))
        pointsArr=contour(n)

        totalArea=0
        count=0
        while (count<n-1):
            xi,yi=pointsArr[count]
            xj,yj=pointsArr[count+1]
            b=xj
            a=xi
            fb=yj
            fa=yi
            curr=(b-a)*((fa+fb)/2)
            totalArea+=curr
            count+=1

        return abs(totalArea)

    
    def fit_shape(self, sample: callable, maxtime: float) -> AbstractShape:
        """
        Build a function that accurately fits the noisy data points sampled from
        some closed shape. 
        
        Parameters
        ----------
        sample : callable. 
            An iterable which returns a data point that is near the shape contour.
        maxtime : float
            This function returns after at most maxtime seconds. 

        Returns
        -------
        An object extending AbstractShape. 
        """

        # replace these lines with your solution

        result = MyShape(sample)
        return result


##########################################################################


import unittest
from sampleFunctions import *
from tqdm import tqdm
import matplotlib.pyplot as plt


class TestAssignment4(unittest.TestCase):
    def test_return(self):
        circ = noisy_circle(cx=1, cy=1, radius=1, noise=0.1)
        ass4 = Assignment4()
        T = time.time()
        shape = ass4.fit_shape(sample=circ, maxtime=5)
        T = time.time() - T
        self.assertTrue(isinstance(shape, AbstractShape))
        self.assertLessEqual(T, 5)

    def test_delay(self):
        circ = noisy_circle(cx=1, cy=1, radius=1, noise=0.1)

        def sample():
            time.sleep(7)
            return circ()

        ass4 = Assignment4()
        T = time.time()
        shape = ass4.fit_shape(sample=sample, maxtime=5)
        T = time.time() - T
        self.assertTrue(isinstance(shape, AbstractShape))
        self.assertGreaterEqual(T, 5)

    def test_circle_area(self):
        circ = noisy_circle(cx=1, cy=1, radius=1, noise=0.1)
        ass4 = Assignment4()
        T = time.time()
        shape = ass4.fit_shape(sample=circ, maxtime=30)
        T = time.time() - T
        a = shape.area()
        self.assertLess(abs(a - np.pi), 0.01)
        self.assertLessEqual(T, 32)

    def test_bezier_fit(self):
        circ = noisy_circle(cx=1, cy=1, radius=1, noise=0.1)
        ass4 = Assignment4()
        T = time.time()
        shape = ass4.fit_shape(sample=circ, maxtime=30)
        T = time.time() - T
        a = shape.area()
        self.assertLess(abs(a - np.pi), 0.01)
        self.assertLessEqual(T, 32)

"""
    def test_dor(self):
        ass4 = Assignment4()
        circ = Circle(cx=1, cy=1, radius=1, noise=0.1)
        xxx=circ.contour
        my_area=ass4.area(xxx,0.001)
        self.assertLess(abs(my_area - np.pi), 0.01)

    def test_circle(self):

        xp = []
        yp = []
        circ=noisy_circle(1,1,1,0.1)
        for i in range(100):
            xi,yi=circ()
            xp.append(xi)
            yp.append(yi)
        fig, (ax12) = plt.subplots(nrows=1, ncols=1, sharex=False, figsize=(12, 4))
        #ax12.plot(xp, yp, color="blue")
        #plt.show()

        ass4=Assignment4()
        newCirc=ass4.fit_shape(circ,200)
        newSampels=newCirc.contour(100)
        xs=[]
        ys=[]
        for xi,yi in newSampels:
            xs.append(xi)
            ys.append(yi)

        ax12.plot(xs,ys,color="red")
        plt.show()
"""


if __name__ == "__main__":
    unittest.main()
