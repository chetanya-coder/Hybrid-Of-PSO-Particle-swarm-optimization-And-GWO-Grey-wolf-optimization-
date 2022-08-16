
import numpy
import math


# define the function blocks

def prod(it):
    p = 1
    for n in it:
        p *= n
    return p


def Ufun(x, a, k, m):
    y = k*((x-a)**m)*(x > a)+k*((-x-a)**m)*(x < (-a))
    return y


def F1(x):
    s = numpy.sum(x**2)
    return s


def F2(x):
    o = sum(abs(x))+prod(abs(x))
    return o


def F3(x):
    dim = len(x)
    o = -20*numpy.exp(-.2*numpy.sqrt(numpy.sum(x**2)/dim)) - \
        numpy.exp(numpy.sum(numpy.cos(2*math.pi*x))/dim)+20+numpy.exp(1)
    return o


def F4(x):
    dim = len(x)
    w = [i for i in range(len(x))]
    w = [i+1 for i in w]
    o = numpy.sum(x**2)/4000-prod(numpy.cos(x/numpy.sqrt(w)))+1
    return o


def F5(x):
    dim = len(x)
    o = .1*((numpy.sin(3*math.pi*x[1]))**2+sum((x[0:dim-2]-1)**2*(1+(numpy.sin(3*math.pi*x[1:dim-1]))**2)) +
            ((x[dim-1]-1)**2)*(1+(numpy.sin(2*math.pi*x[dim-1]))**2))+numpy.sum(Ufun(x, 5, 100, 4))
    return o
