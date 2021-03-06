'''
Forrest Pratson
Final Project
Moving Sofa numerical analysis
'''
from matplotlib import path
import matplotlib.pyplot as plt
import math
import numpy as np
from numpy import argmax
from numpy import asarray
from numpy.random import normal
from numpy.random import random
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from warnings import catch_warnings
from warnings import simplefilter
from matplotlib import pyplot

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
    #Finds the location of each point of the hallway throughout the path
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

'''
Creates a 2d grid of points with a total of 15*res^2 number of points. res must be an integer.
Canvas is a list of tuples, x and y are the individual lists of x and y values
'''
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


'''
Approximates the area of the sofa given the r and t values
N is the number of steps the between 0 and pi which the sofa uses to calculate area
(e.g. if N = 180, then the program will calculate the remaining canvas at each degree)
'''
def appxArea(r,t,canvas,N=100):
    areaC = len(canvas)
    c = canvas.copy()
    aRange = [n0+(pi/N)*i for i in range(N)]
    for a in aRange:
        p = alpha2shape(r,t,a)
        truth = p.contains_points(c)
        c = [c[i] for i in range(len(c)) if truth[i]]
    return(len(c)*(15.0/float(areaC)),c)

'''
Same as appxArea, but only takes one input to allow ease of use when optimizing.
x is a 2 entry list [r,t] 
'''
def bayesFunc(x,N=100):
    #x[0] is r, x[1] is t
    (canvas,who,cares) = createCanvas(N)
    (out,whocares) = appxArea(x[0],x[1],canvas)
    return(out)

'''
Plots the sofa shape by passing the used canvas. Plots a scatter plot of the points included inside the sofa
'''
def canvas2plot(canvas):
    x = [i[0] for i in canvas]
    y = [i[1] for i in canvas]
    plt.axes().set_aspect('equal', 'datalim')
    plt.scatter(x,y,s=.008,marker='x')
    plt.axes().set_aspect('equal', 'datalim')
    plt.title('Maximum Sofa Area Shape')
    plt.savefig('Sofa.png',dpi=500)

'''
Surrogate or approximation for the objective function
Uses gaussian process to predict what the objective function will output
as well as the standard deviation/confidence it has in the value
'''
def surrogate(model, X):
    with catch_warnings():
        simplefilter("ignore")
        return model.predict(X, return_std=True)    

'''
Uses bayesian statistics to find the data point that is most likley to
have a higher objective function value
X is current inputs in model, Xsamamples is a list of possible cannidates, model is the current Gaussian Process model
'''
def acquisition(X, Xsamples, model):
    # calculate the best surrogate score found so far
    yhat, _ = surrogate(model, X)
    best = max(yhat)
    # calculate mean and stdev via surrogate function
    mu, std = surrogate(model, Xsamples)
    mu = mu[:, 0]
    # calculate the probability of improvement
    probs = norm.cdf((mu - best) / (std+1E-9))
    return probs

'''
Passes lots of values to the aquisition function, and finds best canidate for evaluation
returns the best guess of parameters
'''
def opt_acquisition(X, y, model):
    # random search, generate random samples
    Xsamples = random(20000)
    Xsamples = Xsamples.reshape(10000, 2)
    # calculate the acquisition function for each sample
    scores = acquisition(X, Xsamples, model)
    # locate the index of the largest scores
    ix = np.argmax(scores)
    return [Xsamples[ix, 0],Xsamples[ix, 1]]

'''
Finds maximum value of the function two variable, func, using bayesian optimization.

Inputs:
func is the function to be optimized, must take one input only, and its input must be 
and array of the two parameters
rng is the upper and lower bounds of the parameters [xlow,xhigh,ylow,yhigh]
num_it is the number of iterations used to find the maximum value. The more values, the better the guess, but longer
computation time. Morevoer, the function should converge to one point after 20-50 iterations.
init_num is the initial number of samples used to first generate the gaussian model, more than a handful will result in
longer computation time (recomended 5<x<10)

Returns:
maxVal - the maximum value of the function found
paramVal - the values of the parameters of the maximum value
allTrys - lists all of the guessed parameters. Used for creating convergence plot.
'''
def bayesOpt2D(func,rng,num_it,init_num):
    low1 = rng[0][0]
    high1 = rng[0][1]
    r1 = high1-low1
    low2 = rng[1][0]
    high2 = rng[1][1]
    r2 = high2-low2
    X = np.asarray([[(random()+low1)*r1,(random()+low2)*r2] for i in range(init_num)])
    y = np.asarray([func(x) for x in X])
    X = X.reshape(len(X),2)
    y = y.reshape(len(y),1)
    model = GaussianProcessRegressor()
    model.fit(X,y)
    for i in range(num_it):
        x = opt_acquisition(X,y,model)
        actual = func(x)
        est,_ = surrogate(model , [x])
        X = np.vstack((X,[x]))
        y = np.vstack((y,[[actual]]))
        model.fit(X,y)
    allTrys = X
    maxVal = func(X[y.argmax()])
    paramVal = X[y.argmax()]
    return(maxVal,paramVal,allTrys)

'''
Creates a plot of the convergence the bayesian optimization program
plots iterations versus best value found so far.
'''
def convergencePlot(func,params):
    nums = range(1,len(params)+1)
    maxVal = func(params[0])
    y = []
    x = []
    count = 0
    for i in params:
        count += 1
        val = func(i)
        if val > maxVal:
            x.append(count)
            x.append(count)
            y.append(maxVal)
            y.append(val)
            maxVal = val
    x.append(len(params)+1)
    y.append(maxVal)
    plt.xlabel('Number of Iterations')
    plt.ylabel('Best Value')
    plt.title('Convergence Plot')
    plt.plot(x,y,'bo-')
    plt.savefig('Convergence_Plot.png',dpi=500)
    plt.close('all')
    
    
if __name__ == '__main__':
    #test out functions with Hammersly's sofa
    (canvas,x,y) = createCanvas(250)
    realSol = pi/2+2/pi
    hams = appxArea(.5,.5,canvas)[0]
    print("The Approximation of Hamersley's Sofa Area",hams)
    print("Percent Error:",100*abs(hams-realSol)/hams)
    
    
    #Graph the error in the function
    canvasN = range(100,1500,100)
    data = []
    for N in canvasN:
        (canvas,x,y) = createCanvas(N)
        data.append(abs(appxArea(.5,.5,canvas)[0]-realSol))
    plt.plot(canvasN,data)
    plt.title('Error in Area Calculation vs Canvas Size')
    plt.savefig('Sofa_Error_Plot.png',dpi=500)
    plt.close('all')
    
    #Use Bayesian Optimization to find max area and plot converges
    bay = bayesOpt2D(bayesFunc,[(0,1),(0,1)],150,25)
    print('The maximum area found was:',bay[0])
    convergencePlot(bayesFunc,bay[-1])
    
    #Plot Sofa figure from Bayesian Optimization
    (canvas,x,y) = createCanvas(500)
    (area,c) =appxArea(bay[1][0],bay[1][1],canvas)
    canvas2plot(c)
    plt.close('all')
    
    #Plot 3d plot of values
    plt.close('all')
    (canvas,x,y) = createCanvas(100)
    x = y = np.arange(0, 1.4, 0.1)
    X, Y = np.meshgrid(x, y)
    zs = []
    for i in range(len(np.ravel(X))):
        zs.append(appxArea(np.ravel(X)[i],np.ravel(Y)[i],canvas)[0])
    zs = np.array(zs)
    Z = zs.reshape(X.shape)
    plt.contourf(X, Y, Z,15)
    plt.colorbar();
    plt.title('Contour Plot of Sofa Area')
    plt.xlabel('r Value')
    plt.ylabel('t Value')
    plt.savefig('Contour_Plot.png',dpi=1000)
    plt.close('all')
