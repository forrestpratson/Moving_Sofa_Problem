'''
Forrest Pratson
Final Project
Moving Sofa numerical analysis
'''
from matplotlib import path
import matplotlib.pyplot as plt
import math
import numpy as np
from skopt import gp_minimize
from skopt.plots import plot_convergence
from mpl_toolkits.mplot3d import Axes3D

'''
define common trig functions and values for ease of use
'''
def cos(x):
    return(math.cos(x))

def tan(x):
    return(math.tan(x))

def sin(x):
    return(math.sin(x))

r2 = math.sqrt(2)
pi = math.pi
n0 = 10**-16

'''
Takes the r,t, and alpha, and finds the path of the shape
'''
def alpha2shape(r,t,a,plot=False):
    A0 = (r*cos(a),t*sin(a))
    A1 = (r*cos(a)+r2*cos(pi/4+a/2),t*sin(a)+r2*sin(pi/4+a/2))
    B0 = (r*cos(a)-(t*sin(a))/(tan(a/2)),0)
    B1 = (r*cos(a)-(t*sin(a))/(tan(a/2))-1/(sin(a/2)),0)
    C0 = (r*cos(a)+t*sin(a)*tan(a/2),0)
    C1 = (r*cos(a)+t*sin(a)*tan(a/2)+1/(cos(a/2)),0)
    p = path.Path([A0,C0,C1,A1,B1,B0])
    if plot == True:
        x = [A0[0],C0[0],C1[0],A1[0],B1[0],B0[0],A0[0]]
        y = [A0[1],C0[1],C1[1],A1[1],B1[1],B0[1],A0[1]]
        return(p,x,y)
    else:
        return(p)

def createCanvas(res):
    d = 1.0/float(res)
    x = []
    for i in range(res+1):
        for whocares in range(res+1):
            x.append(-3+i*d*6)
    y = [i*2.5*d for i in range(res+1)]*(res+1)
    mx = np.matrix([x,y]).T
    canvas = [(mx[i,0],mx[i,1]) for i in range(len(x))]
    return(canvas,x,y)

def appxArea(r,t,canvas,N=100,plot=False,x=None,y=None):
    areaC = len(canvas)
    c = canvas.copy()
    aRange = [n0+(pi/N)*i for i in range(N)]
    for a in aRange:
        p = alpha2shape(r,t,a)
        truth = p.contains_points(c)
        c = [c[i] for i in range(len(c)) if truth[i]]
    return(len(c)*(15.0/float(areaC)),c)

def bayesFunc(x,N=100):
    #x0 is r, x1 is t
    (canvas,who,cares) = createCanvas(N)
    (out,whocares) = appxArea(x[0],x[1],canvas)
    return(-1*out)

def canvas2plot(canvas):
    x = [i[0] for i in canvas]
    y = [i[1] for i in canvas]
    plt.axes().set_aspect('equal', 'datalim')
    plt.scatter(x,y,s=.008,marker='x')
    plt.axes().set_aspect('equal', 'datalim')
    plt.savefig('Sofa.png',dpi=500)
    
if __name__ == '__main__':
     #test out functions with Hammersly's sofa
    (canvas,x,y) = createCanvas(50)
    print("The Approximation of Hamersley's Sofa Area",appxArea(.5,.5,canvas)[0])
    
    #Graph the error in the function
    realSol = pi/2+2/pi
    canvasN = range(100,1500,100)
    data = []
    for N in canvasN:
        (canvas,x,y) = createCanvas(N)
        data.append(abs(appxArea(.5,.5,canvas)[0]-realSol))
    plt.plot(canvasN,data)
    plt.savefig('Sofa_Error_Plot.png',dpi=500)
    plt.close('all')
