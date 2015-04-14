from partie1 import *

#determines position and speed by both axis of planet B, A being the sun at supposed immobile at (0,0)
def position (mA,mB,y0,tf,eps,meth) :
    G = 1
    F = (4,0.,y0,lambda y,t : np.array([y[1],(-G*mA*mB*y[0])/pow((pow(y[0],2) + pow(y[2],2)),3/2.),y[3],(-G*mA*mB*y[2])/pow((pow(y[0],2) + pow(y[2],2)),3/2.)]))

    R = meth_epsilon(F[2],F[1],tf,eps,F[3],meth,F[0])

    return R

#different tests

y0 = np.array([1.,0.,0.,0.1])
## y1 = np.array([1.,-0.1,0.,0.1])
## y2 = np.array([1.,-10.,0.,0.1])
cercle = position(1,0.01,y0,75,0.1,runge_kutta)
## diverge = position(1,0.01,y1,30,0.1,runge_kutta)
## converge = position(1,0.01,y2,0.25,0.01,runge_kutta)


#affiche trajectoire a partir de position
def plotpos(R) :
    X = np.zeros(R.shape[0])
    Y = np.zeros(R.shape[0])
    D =  np.zeros(R.shape[0])
    for i in range (R.shape[0]):
        X[i] = R[i,0]
        Y[i] = R[i,2]
        D[i] = np.sqrt(pow(X[i],2) + pow(Y[i],2))
    mp.plot(X,Y,'b',label =  "terre")
    mp.plot(0,0,'ys',label = "soleil")


#allows to compare 3 different trajectories if given position
def temp(R,S,T) :
    X = np.zeros(R.shape[0])
    Y = np.zeros(R.shape[0])
    D =  np.zeros(R.shape[0])
    for i in range (R.shape[0]):
        X[i] = R[i,0]
        Y[i] = R[i,2]
        D[i] = np.sqrt(pow(X[i],2) + pow(Y[i],2))
    mp.plot(X,Y,'b',label = "y0 = [1.,0,0.,0.1]")
    
    X1 = np.zeros(S.shape[0])
    Y1 = np.zeros(S.shape[0])
    D1 =  np.zeros(S.shape[0])
    for i in range (S.shape[0]):
        X1[i] = S[i,0]
        Y1[i] = S[i,2]
        D1[i] = np.sqrt(pow(X1[i],2) + pow(Y1[i],2))
    mp.plot(X1,Y1,'r',label = "y0 = [1.,-0.1,0.,0.1]")

    X2 = np.zeros(T.shape[0])
    Y2 = np.zeros(T.shape[0])
    D2 =  np.zeros(T.shape[0])
    for i in range (T.shape[0]):
        X2[i] = T[i,0]
        Y2[i] = T[i,2]
        D2[i] = np.sqrt(pow(X2[i],2) + pow(Y2[i],2))
    mp.plot(X2,Y2,'k',label = "y0 = [1.,-10.,0.,0.1]")
    
    mp.plot(0,0,'ys',label = "soleil")
    mp.xlabel("x")
    mp.ylabel("y")
    mp.title("Trajectoire de la Terre autour du soleil avec differentes positions initiales")
    mp.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
        fancybox=True, shadow=True, ncol=5)
    mp.show()



#gives position and speed by both axis of meteorite C, B being the earth having a circular trajectory around the sun A fixed at (0,0). Initial position of earth (1,0)
def position3c (mA,mB,mC,y0,tf,eps,meth) :
    G = 1
    F = (4,0.,y0,lambda y,t : np.array([y[1],((-G*mA*mC*y[0])/pow((pow(y[0],2) + pow(y[2],2)),3/2.)) - ((G*mB*mC*(y[0] - np.cos(t)))/pow((pow(y[0]-np.cos(t),2) + pow(y[2]-np.sin(t),2)),3/2.)) ,y[3],((-G*mA*mC*y[2])/pow((pow(y[0],2) + pow(y[2],2)),3/2.)) - ((G*mB*mC*(y[2] - np.sin(t)))/pow((pow(y[0]-np.cos(t),2) + pow(y[2]-np.sin(t),2)),3/2.))]))

    R = meth_epsilon(F[2],F[1],tf,eps,F[3],meth,F[0])

    return R



y0m = np.array([0.99,0.,0.,0.005])

#plots trajectory of meteorite from position
def plotpos3c(R) :
    X = np.zeros(R.shape[0])
    Y = np.zeros(R.shape[0])
    D =  np.zeros(R.shape[0])
    for i in range (R.shape[0]):
        X[i] = R[i,0]
        Y[i] = R[i,2]
        D[i] = np.sqrt(pow(X[i],2) + pow(Y[i],2))
    plotpos(cercle)
    mp.plot(X,Y,'r',label = "meteorite")
    mp.xlabel("x")
    mp.ylabel("y")
    mp.title("Trajectoire de la Terre et d'une meteorite autour du soleil")
    mp.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
        fancybox=True, shadow=True, ncol=5)
    mp.show()
