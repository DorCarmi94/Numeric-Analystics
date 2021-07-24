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
from functionUtils import AbstractShape


class MyShape(AbstractShape):
    # change this class with anything you need to implement the shape
    def __init__(self,shapeLineFunctions,firstAndLastDotForEveryLine):
        pass

    def sample(self):
        return 0
    def contour(self, n: int):
        return 0

    def area(self) -> np.float32:
        return 0


class Assignment4:
    def __init__(self):
        """
        Here goes any one time calculation that need to be made before 
        solving the assignment for specific functions. 
        """

        pass

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

        # array of pairs= (m,n)
        funcs=[]

        #start,end -> paris of xs
        startsEnds=[]

        for i in range(100):
            samplesArr = []
            Xs = []
            Ys=[]
            for i in range(10):
                sam=sample()
                x,y=sam
                samplesArr.append(sam)
                Xs.append(x)
                Ys.append(y)
            leng=len(samplesArr)
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

            for i in range (leng):
                B=[1]*leng
                mmHorizental_At=np.matrix([Xs,B])
                mmVertical_A=mmHorizental_At.T
                AtA=mmHorizental_At*mmVertical_A

                Atb=mmHorizental_At*Ys

                solve=np.linalg(AtA,Atb)
                m=solve[0,0]
                n=solve[0,1]
                funcs.append((m,n))








        #result = MyShape()
        #x, y = sample()

        return 0


##########################################################################


import unittest
from sampleFunctions import *
from tqdm import tqdm
import matplotlib.pyplot as plt

class TestAssignment4(unittest.TestCase):

    def test_return(self):

        fig, (ax12) = plt.subplots(nrows=1, ncols=1, sharex=False, figsize=(12, 4))

        circ = noisy_circle(cx=1, cy=1, radius=1, noise=0.1)
        ass4 = Assignment4()
        xs=[]
        ys=[]
        for i in range(100):
            xi,yi=circ()
            xs.append(xi)
            ys.append(yi)
        markers_on=xs
        plt.plot(xs, ys, '-gD', markevery=markers_on)
        ax12.plot(xs, ys, color="blue")
        plt.show()

        T = time.time()
        shape = ass4.fit_shape(sample=circ, maxtime=5)
        T = time.time() - T
        self.assertTrue(isinstance(shape, AbstractShape))
        self.assertLessEqual(T, 5)

    """
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

if __name__ == "__main__":
    unittest.main()

    def __init__(self, f:callable):
        pass

        #init
        states = ["forwardInit","forward", "backward"]
        currentState = states[0]
        self._fSamples=f

        numberofSamples=100
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


        ass4A = Assignment4A()



        #state machine

        while (counter<numberofSamples):
            prevXi, prevYi = newXi, newYi
            lastXi = newXi
            newXi, newYi = self._fSamples()

            if(currentState=="forwardInit"):
                if(newXi<prevXi):
                    currentState="backward"

                    #-----------
                    def h_forward(x):
                        prevYf = np.float32(0)
                        for xj, yj in currSamples:
                            if(x>=xj):
                                prevYf=yj
                            else:
                                return prevYf
                        return float("nan")
                    #--------
                    newG=ass4A.fit(h_forward,firstXi,lastXi,len(currSamples),20)
                    currSamples=[]
                    self._functions.append(newG)
                    self._xStratXendForEachFunc.append(firstXi,lastXi)
                    firstXi=newXi

            elif (currentState == "backward"):
                if(newXi>prevXi):
                    currentState="forward"

                    # -----------
                    def h_backward(x):
                        for xj, yj in currSamples:
                            if (x < xj):
                                continue
                            else:
                                return yj
                        return float("nan")

                    # --------
                    newG = ass4A.fit(h_backward, firstXi, lastXi, len(currSamples), 20)
                    currSamples = []
                    self._functions.append(newG)
                    self._xStratXendForEachFunc.append(firstXi, lastXi)
                    firstXi = newXi

            counter+=1

            #lastOne:
            # -----------
        def h_backward(x):
            for xj, yj in currSamples:
                if (x < xj):
                    continue
                else:
                    return yj
            return float("nan")

        # --------
        newG = ass4A.fit(h_backward, firstXi, lastXi, len(currSamples), 20)
        self._functions.append(newG)
        self._xStratXendForEachFunc.append(firstXi, lastXi)