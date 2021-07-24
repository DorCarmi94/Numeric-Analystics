"""
In this assignment you should fit a model function of your choice to data 
that you sample from a given function. 

The sampled data is very noisy so you should minimize the mean least squares 
between the model you fit and the data points you sample.  

During the testing of this assignment running time will be constrained. You
receive the maximal running time as an argument for the fitting method. You 
must make sure that the fitting function returns at most 5 seconds after the 
allowed running time elapses. If you take an iterative approach and know that 
your iterations may take more than 1-2 seconds break out of any optimization 
loops you have ahead of time.

Note: You are NOT allowed to use any numeric optimization libraries and tools 
for solving this assignment. 

"""

import numpy as np
import time
import random

from NewVersion.functionUtils import NOISY, DELAYED


class Assignment4A:
    def __init__(self):
        """
        Here goes any one time calculation that need to be made before 
        solving the assignment for specific functions. 
        """

        pass
    def isNaN(self,num):
        return num!=num

    def fit(self, f: callable, a: float, b: float, d:int, maxtime: float) -> callable:
        """
        Build a function that accurately fits the noisy data points sampled from
        some closed shape. 
        
        Parameters
        ----------
        f : callable. 
            A function which returns an approximate (noisy) Y value given X. 
        a: float
            Start of the fitting range
        b: float
            End of the fitting range
        d: int 
            The expected degree of a polynomial matching f
        maxtime : float
            This function returns after at most maxtime seconds.

        Returns
        -------
        a function:float->float that fits f between a and b
        """
        def badFunc(x):
            return float("nan")

        #getAllSamples
        startTime=time.time()
        n=np.float32(10*d)
        xs=[]
        ys=[]
        points=[]
        stepSize=np.float32((b-a)/n)
        currentX=np.float32(a)
        currentTime=startTime
        deltaTime=time.time()-startTime
        i=0
        while (i<n and deltaTime<maxtime):
            xs.append(currentX)
            fx=np.float32(f(currentX))
            ys.append(fx)
            points.append((currentX,fx))
            currentX=currentX+stepSize
            deltaTime=time.time()-startTime
            i+=1


        #create the matrix and calc
        memoization=[float("nan")]*(2*d+1)
        fullXsMatrix=[]


        #Ax=B
        #Calc A:
        i=0
        while(i<d+1):
            lineI=[]
            if(time.time()-startTime>maxtime):
                return badFunc
            j=0
            while (j<d+1):
                if (time.time() - startTime > maxtime):
                    return badFunc
                if(self.isNaN(memoization[j+i])):
                    sum=0
                    p=0
                    while (p<n):
                        sum=sum+np.power(xs[p],j+i)
                        p+=1
                    memoization[j+i]=sum
                lineI.append(memoization[j+i])
                j+=1
            fullXsMatrix.append(lineI)
            i+=1

        #Calc B:
        fullYsVector=[]
        i=0
        while (i<d+1):
            if (time.time() - startTime > maxtime):
                return badFunc
            sum=0
            p=0
            while(p<n):
                if (time.time() - startTime > maxtime):
                    return badFunc
                sum=sum+ys[p]*np.power(xs[p],i)
                p+=1
            fullYsVector.append(sum)
            i+=1


        #convert to matrix and make matrix calculations

        Amatrix=np.matrix(fullXsMatrix)
        AmatrixT=Amatrix
        AmatrixInv=np.linalg.inv(AmatrixT)
        Bmatrix=np.matrix(fullYsVector)
        BmatrixTrans=Bmatrix.T

        #finally calc the arguments
        argumnetsVector=AmatrixInv*BmatrixTrans

        anotherTry2=np.linalg.solve(Amatrix,BmatrixTrans)

        def g(x):
            theAns=0
            for i in range (d+1):
                theAns=theAns+argumnetsVector[i]*np.power(x,i)
            return theAns

        return g



##########################################################################


import unittest
from sampleFunctions import *
from tqdm import tqdm


class TestAssignment4(unittest.TestCase):

    def test_dor(self):
        ass4=Assignment4A()
        f=np.poly1d([1,1,1])
        g=ass4.fit(f,-2,1,2,1800)
    def test_return(self):
        f = NOISY(0.01)(poly(1,1,1))

        ass4 = Assignment4A()
        T = time.time()
        shape = ass4.fit(f=f, a=0, b=1, d=10, maxtime=1800)
        T = time.time() - T
        self.assertLessEqual(T, 5)

    def test_delay(self):
        f = DELAYED(7)(NOISY(0.01)(poly(1,1,1)))

        ass4 = Assignment4A()
        T = time.time()
        shape = ass4.fit(f=f, a=0, b=1, d=10, maxtime=5)
        T = time.time() - T
        self.assertGreaterEqual(T, 5)

    def test_err(self):
        f = poly(1,1,1)
        nf = NOISY(1)(f)
        ass4 = Assignment4A()
        T = time.time()
        ff = ass4.fit(f=nf, a=0, b=1, d=10, maxtime=5)
        T = time.time() - T
        mse=0
        for x in np.linspace(0,1,1000):            
            self.assertNotEquals(f(x), nf(x))
            mse+= (f(x)-ff(x))**2
        mse = mse/1000
        print(mse)


if __name__ == "__main__":
    unittest.main()
