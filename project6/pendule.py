from partie1 import *

from decimal import Decimal


#calcul la frequence du pendule
def freq(y0,t0,tf,eps,f,meth,dim):
    print "angle initial :"
    print y0[0]
    print "calcul trajectoire :"
    A = meth_epsilon(y0,t0,tf,eps,f,meth,dim)
    print ""
    mx = A.shape[0]
    tf=float(tf)
    dt=float(tf/mx)
    t=float(dt)
    b1=mt.copysign(1,A[0,1])
    for i in range (1,(mx-1)):
        b2=mt.copysign(1,A[i,1])
        df=b1+b2
        if (df==0):
            print "freq :"
            df=df/2
            t=t+df
            t=t+t
            t=1./t
            print t
            print ""
            return t
        t=t+dt


#Calcul de la frequence en fonction des condition initial
def pltfrqp(N,scale):
    y=[]
    x=[]
    for i in np.arange (-N,N+scale,scale):
        if (abs(i)>scale*scale):
            f4=(2,0.,np.array([i,0]),lambda y,t : np.array([y[1],-10*np.sin(y[0])]) )
            y.append(freq(f4[2],f4[1],3.4,0.1 ,f4[3],runge_kutta,f4[0]))
            x.append(i)
    return x,y
        







#f4  =(2,0.,np.array([1.,0]),lambda y,t : np.array([y[1],-10*np.sin(y[0])]))

# # ## f5 = (2,0.,np.array([1,1]),lambda y,t : np.array([2*y[0],-y[1]]))

# A= meth_epsilon(f4[2],f4[1],4  ,1 ,f4[3],step_euler,f4[0])

#x,y=pltfrqp(3.1,0.1)
#print x
#print y


#calcul de pltfrqp(0.2,0.01)
#x=[-0.19, -0.17999999999999999, -0.16999999999999998, -0.15999999999999998, -0.14999999999999997, -0.13999999999999996, -0.12999999999999995, -0.11999999999999994, -0.10999999999999993, -0.099999999999999922, -0.089999999999999913, -0.079999999999999905, -0.069999999999999896, -0.059999999999999887, -0.049999999999999878, -0.039999999999999869, -0.02999999999999986, -0.019999999999999851, -0.0099999999999998423, 0.010000000000000175, 0.020000000000000184, 0.030000000000000193, 0.040000000000000202, 0.050000000000000211, 0.06000000000000022, 0.070000000000000229, 0.080000000000000238, 0.090000000000000246, 0.10000000000000026, 0.11000000000000026, 0.12000000000000027, 0.13000000000000028, 0.14000000000000029, 0.1500000000000003, 0.16000000000000031, 0.17000000000000032, 0.18000000000000033, 0.19000000000000034]
#y=[0.5021568777911706, 0.5022733817713487, 0.5023835006127568, 0.5024872258797726, 0.5025848887996517, 0.5026758036740454, 0.5027606419829905, 0.5028393975560475, 0.5029120646623859, 0.502977958613137, 0.5030377536232796, 0.5030921250618933, 0.5031397097129714, 0.5031805037220289, 0.5032158638806096, 0.5032430673849441, 0.5032702738296628, 0.503281157232246, 0.5033029254482037, 0.5033029254482037, 0.503281157232246, 0.5032702738296628, 0.5032430673849441, 0.5032158638806096, 0.5031805037220289, 0.5031397097129714, 0.5030921250618933, 0.5030377536232796, 0.502977958613137, 0.5029120646623859, 0.5028393975560475, 0.5027606419829905, 0.5026758036740454, 0.5025848887996517, 0.5024872258797726, 0.5023835006127568, 0.5022733817713487, 0.5021568777911706]




x=[-3.1000000000000001, -3.0, -2.8999999999999999, -2.7999999999999998, -2.6999999999999997, -2.5999999999999996, -2.4999999999999996, -2.3999999999999995, -2.2999999999999994, -2.1999999999999993, -2.0999999999999992, -1.9999999999999991, -1.899999999999999, -1.7999999999999989, -1.6999999999999988, -1.5999999999999988, -1.4999999999999987, -1.3999999999999986, -1.2999999999999985, -1.1999999999999984, -1.0999999999999983, -0.99999999999999822, -0.89999999999999813, -0.79999999999999805, -0.69999999999999796, -0.59999999999999787, -0.49999999999999778, -0.39999999999999769, -0.2999999999999976, -0.19999999999999751,  0.20000000000000284, 0.30000000000000293, 0.40000000000000302, 0.50000000000000311, 0.6000000000000032, 0.70000000000000329, 0.80000000000000338, 0.90000000000000346, 1.0000000000000036, 1.1000000000000036, 1.2000000000000037, 1.3000000000000038, 1.4000000000000039, 1.500000000000004, 1.6000000000000041, 1.7000000000000042, 1.8000000000000043, 1.9000000000000044, 2.0000000000000044, 2.1000000000000045, 2.2000000000000046, 2.3000000000000047, 2.4000000000000048, 2.5000000000000049, 2.600000000000005, 2.7000000000000051, 2.8000000000000052, 2.9000000000000052, 3.0000000000000053, 3.1000000000000054]
y=[0.1503056606521207, 0.19574199981040985, 0.22514163695556383, 0.24906261005945227, 0.26997783228898026, 0.28889275082328775, 0.3063351545916159, 0.3225969050860029, 0.33786904934742157, 0.35225318197460925, 0.36586062996633895, 0.37874304651444873, 0.39091489355211184, 0.40242713868023583, 0.4132951896315292, 0.4235479419872412, 0.43314800464287956, 0.4421523145638126, 0.450539239151936, 0.4583000648065704, 0.46546808634440756, 0.4720042480383105, 0.47793650912190866, 0.4832741825870691, 0.4879722465785175, 0.49198979121179665, 0.49548642831701234, 0.4985044865403894, 0.5006257822277906, 0.5027652086475575, 0.5027652086475575, 0.5006257822277906, 0.4985044865403894, 0.49548642831701234, 0.49198979121179665, 0.4879722465785175, 0.4832741825870691, 0.47793650912190866, 0.4720042480383105, 0.46546808634440756, 0.4583000648065704, 0.450539239151936, 0.4421523145638126, 0.43314800464287956, 0.4235479419872412, 0.4132951896315292, 0.40242713868023583, 0.39091489355211184, 0.37874304651444873, 0.36586062996633895, 0.35225318197460925, 0.33786904934742157, 0.3225969050860029, 0.3063351545916159, 0.28889275082328775, 0.26997783228898026, 0.24906261005945227, 0.22514163695556383, 0.19574199981040985, 0.1503056606521207]





z=[]
for i in range(len(x)):
    z.append(0.5033)



mp.plot(x, y, color="blue", linewidth=1.5)
mp.plot(x, z, color="red",  linewidth=2.5, linestyle="-", label="$y=((g/l)^{1/2})/(2*\pi)$")
mp.grid()
mp.xlabel("Angle Initial ($rad$)")
mp.ylabel("Frequence ($Hz$)")
mp.ylim(y[0],0.60)
mp.legend(loc='upper right')
mp.show()


# fonction renvoie un tableaux des angles et vitesses angulaires du pendule double 
def pendule (m1,m2,l1,l2,y0,tf,g,eps,meth) :
    p = (4,0.,y0,lambda y,t : np.array([y[1],(-g*(2*m1+m2)*np.sin(y[0])-m2*g*np.sin(y[0]-2*y[2])-2*np.sin(y[0]-y[2])*m2*((pow(y[3],2))*l2+(pow(y[1],2))*l1*np.cos(y[0]-y[2])))/(l1*(2*m1+m2-m2*np.cos(2*y[0]-2*y[2]))),y[3],(2*np.sin(y[0]-y[2])*((pow(y[1],2))*l1*(m1+m2)+g*(m1+m2)*np.cos(y[0])+(pow(y[3],2))*l2*m2*np.cos(y[0]-y[2])))/(l2*(2*m1+m2-m2*np.cos(2*y[0]-2*y[2])))]))

    R = meth_epsilon(p[2],p[1],tf,eps,p[3],meth,p[0])    
    return R

# fonction qui renvoie la position catesienne de lextremite du pendule
def extremite(theta1,theta2,l1,l2):
    x=l1*np.sin(theta1)+l2*np.sin(theta2)
    y=-l1*np.cos(theta1)-l2*np.cos(theta2)
    return [x,y]


def plotpen (m1,m2,l1,l2,y0,tf,g,eps,meth) :
    R = pendule (m1,m2,l1,l2,y0,tf,g,eps,meth)
    X = np.zeros(R.shape[0])
    Y = np.zeros(R.shape[0])
    for i in range (R.shape[0]):
        tmp = extremite(R[i,0],R[i,2],l1,l2)
        X[i] = tmp[0]
        Y[i] = tmp[1]
    return X,Y


#############
#trace de 2 trajectoires de condition initial proche

# y01 = np.array([2.,0,2.,0])
# y02 = np.array([2.1,0,2.,0])

# X,Y=plotpen (1.,1.,1.,1.,y01,5.,10.,5.,runge_kutta)
# Z,T=plotpen (1.,1.,1.,1.,y02,5.,10.,5.,runge_kutta)

# mp.plot(X,Y,color="red", linewidth=1.,label="$y0=[2.,0,2.,0]$")
# mp.plot(Z,T,color="blue", linewidth=1.,label="$y0=[2.1,0,2.,0]$")
# mp.grid()
# mp.xlabel("Axe des $x$")
# mp.ylabel("Axe des $z$")
# mp.ylim(-2.,2.)
# mp.legend(loc='upper right')

# mp.show()



# donne le temps de retournement pour divert parametre
def retournement(m1,m2,l1,l2,y0,tf,g,eps,meth) :
     R = pendule (m1,m2,l1,l2,y0,tf,g,eps,meth)
     mx = R.shape[0]
     dt=tf/mx
     t=0
     for i in range (1,(mx-1)) :
        t=t+dt
        if (np.cos(R[i,0])<0):
            if (np.sin(R[i,0]) >0 ):
                if(np.sin(R[i+1,0]) < 0):
                    df2=dt/2
                    t=t+df2
                    return t
            if (np.sin(R[i,0]) < 0):
                if(np.sin(R[i+1,0]) > 0):
                    df2=dt/2
                    t=t+df2
                    return t
        if (np.cos(R[i,2])<0):
            if (np.sin(R[i,2]) >0 ):
                if(np.sin(R[i+1,2]) < 0):
                    df2=dt/2
                    t=t+df2
                    return t
            if (np.sin(R[i,2]) < 0):
                if(np.sin(R[i+1,2]) > 0):
                    df2=dt/2
                    t=t+df2
                    return t
     t=0
     return t






import multiprocessing
pool = multiprocessing.Pool()

#renvoie le temps de retournement pour une condition initial
def retournement_op(y0):
    print y0
    return retournement(1.,1.,1.,1.,y0,10.,10.,5.,runge_kutta)
 

#calcul la matrice qui fait correspondre condition initial et temps de retournement
def map_retournement(scale):
    imax=(int(3.1/scale))*2
    jmax=(int(3.1/scale))*2
    memoire=[]
    M=np.zeros([imax,imax])
    print M
    im=int(imax/2)
    jm=int(jmax/2)
    Ly0=[]
    for i in np.arange(-3.1,3.1,scale):
        if (abs(i)<3.14) and (abs(i)>scale*scale):
            for j in  np.arange(-3.1,3.1,scale):
                if (abs(j)<3.14) and (abs(j)>scale*scale):
                    Ly0.append(np.array([i,0,j,0]))
    print Ly0

    result=[]
    result=map(retournement_op,Ly0)
    print result

    t=result.pop()
    print t
    for i in np.arange(-3.1,3.1,scale):
        if (abs(i)<3.14) and (abs(i)>scale*scale):
            for j in  np.arange(-3.1,3.1,scale):
                if (abs(j)<3.14) and (abs(j)>scale*scale):
                    I=Ly0.pop()
                    x=I[0]
                    y=I[2]
                    print x,y
                    x=int((Decimal(str(round(x,1))))*(Decimal(str(round((1./scale),1))))+im)
                    y=int((Decimal(str(round(y,1))))*(Decimal(str(round((1./scale),1))))+jm)
                    print x,y
                    M[x,y]=t

    return M




scale = 1.


#ne fonction pas encore de la facon souhaite
#M=map_retournement(scale)
#mp.show(mp.imshow(M))


##tentative de comprehension du multiprocessing

# L=[]

# for i in range(1000):
#     L.append(i)

# print L

# def fonc (t):
#     return t*t


# t=2

# print t

# print fonc(t)

# LL=pool.map(fonc,L)

# print LL
