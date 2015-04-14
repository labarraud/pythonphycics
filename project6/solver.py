import numpy as np
import matplotlib.pyplot as mp;
from math import *
import math as mt
import time, sys
from pylab import *

#different step methods

def step_euler(f,t,h,y):
    return y+h*f(y,t)

def step_middle(f,t,h,y):
    y05=y+(h/2)*f(y,t)
    pn=f(y05,t+h/2)
    yn1=y +pn*h
    return yn1

def step_heun(f,t,h,y):
    pn1=f(y,t)
    yn2=y+h*f(y,t)
    pn2=f(yn2,t+h)
    yn1=y+(h/2)*(pn1+pn2)
    return yn1

def runge_kutta(f,t,h,y):
    pn1=f(y,t)
    yn2=y+(h/2)*f(y,t)
    pn2=f(yn2,t+h/2)
    yn3=y+(h/2)*pn2
    pn3=f(yn3,t+h/2)
    yn4=y+h*pn3
    pn4=f(yn4,t+h)
    return y+(h/6)*(pn1+2*pn2+2*pn3+pn4)



def meth_n_step(y0,t0,tf,N,f,meth,dim):
    t = t0
    h = (tf - t0)/N
    y = np.zeros([N,dim])
    y[0]= meth(f,t,h,y0)
    for i in range(1,N):
        y[i]= meth(f,t,h,y[i-1])
        t = t+h
    return y


def meth_epsilon(y0,t0,tf,eps,f,meth,dim):
    N = 100
    y1 = meth_n_step(y0,t0,tf,N,f,meth,dim)
    y2 = meth_n_step(y0,t0,tf,2*N,f,meth,dim)
    var=np.max(np.linalg.norm(y2[0:2*N:2] - y1[0:N]))
    while (var > eps) :
        N = 2*N
        y1 = y2
        y2 = meth_n_step(y0,t0,tf,2*N,f,meth,dim)
        var=np.max(np.linalg.norm(y2[0:2*N:2] - y1[0:N]))
    return y2



def freq(y0,t0,tf,eps,f,meth,dim):
    A = meth_epsilon(y0,t0,tf,eps,f,meth,dim)
    max = A.shape[0]
    dt=tf/max
    t=dt
    for i in range (1,(max-1)):
        t=t+dt
        if(A[0,0] - A[i,0] < 0.000001):
                return t

def plotfreq(y0,t0,tf,eps,f,meth,dim):
    T = freq(y0,t0,tf,eps,f,meth,dim)
    A = meth_epsilon(y0,t0,tf,eps,f,meth,dim)
    y=np.zeros([A.shape[0],A.shape[1]])
    for i in range (0,A.shape[0]):
        y[i]=T*(1+(pow(A[i,0],2))/16)
    return y

#function is list : f = (dimension,t0,y(t0),y'(t)=f(y))

#tests :

f1 = (1,0.,1.,lambda y,t: y/(1+pow(t,2)))

sol1 = lambda t : exp(np.arctan(t))

f2 = (2,0.,np.array([1,0]),lambda y,t: np.array([-y[1],y[0]]))

sol2 = lambda t : np.array([cos(t),sin(t)])

## f4 = (2,0.,np.array([3.14/4,0]),lambda y,t : np.array([y[1],-10*y[0]]))

## f5 = (2,0.,np.array([1,1]),lambda y,t : np.array([2*y[0],-y[1]]))

## A= meth_epsilon(f4[2],f4[1],4  ,1 ,f4[3],step_euler,f4[0])

## F=freq(f4[2],f4[1],4.00  ,1 ,f4[3],step_euler,f4[0])




#champs des tangentes pour eq 2
X,Y = meshgrid( arange(-4.,4.,.3),arange(-2.,2,.3) )
U = 1
V = -sin(X)

#1
figure()

quiver(X,Y, U, V)
title('Champs des tangentes pour eq 2 ')
axis([-4.5, 4.5, -2.5, 2.5])


show()

#compares different step methods between them and with the solutions. E is the equation you want to solve and sol it's solution.
def comparaison(E,tf,eps,sol) :
     Euler = meth_epsilon(E[2],E[1],tf,eps,E[3],step_euler,E[0])
     Middle = meth_epsilon(E[2],E[1],tf,eps,E[3],step_middle,E[0])
     Heun = meth_epsilon(E[2],E[1],tf,eps,E[3],step_heun,E[0])
     Kutta = meth_epsilon(E[2],E[1],tf,eps,E[3],runge_kutta,E[0])

     Te = np.zeros(Euler.shape[0])
     Tm = np.zeros(Middle.shape[0])
     Th = np.zeros(Heun.shape[0])
     Tk = np.zeros(Kutta.shape[0])
     Sol = np.zeros([Euler.shape[0],E[0]])     
     k = 0

     for i in np.arange (E[1],tf,(tf-E[1])/Euler.shape[0]) :
         Te[k] = i
         Sol[k] = sol(i) 
         k = k+1

     k = 0
     for i in np.arange (E[1],tf,(tf-E[1])/Middle.shape[0]) :
         Tm[k] = i
         k = k+1

     k = 0
     for i in np.arange (E[1],tf,(tf-E[1])/Heun.shape[0]) :
         Th[k] = i
         k = k+1

     k = 0
     for i in np.arange (E[1],tf,(tf-E[1])/Kutta.shape[0]) :
         Tk[k] = i
         k = k+1
         
     eul= mp.plot(Te,Euler,'b',label = "Euler")
     mid= mp.plot(Tm,Middle,'r', label = "Middle Point")
     heun= mp.plot(Th,Heun,'m', label =" Heun ")
     kutt= mp.plot(Tk,Kutta, 'y', label = " Kutta ")
     sol= mp.plot(Te,Sol,'k', label = " Solution ")
     mp.xlabel (" t")
     mp.ylabel ("y=f(t)")
     mp.title(" Comparaison des methodes pour l equation 2 " )
     mp.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
          fancybox=True, shadow=True, ncol=5)
     mp.show()
     
     


