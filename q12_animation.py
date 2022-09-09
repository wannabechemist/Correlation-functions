#program that animates q12 correlation functions derived in:
#screening of ionic interactions in a charged adsorbent - 
#the application of Replica Ornstein - Zernike equations 
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
zp=1 #this is the charge number of cations - annealed fluid
z0=1 #this is the charge number of cations - matrix
z0d=1 #this is the charge number of anions - template
rmax=60 #this is the radius or the size of the x axis
r=np.arange(0.1,rmax,0.1)
ymax=1.5 #this is the value of q12 functions or the size of the y axis
Q=1.2
epsilon=1
#c0=0.425
c0=0.425
#c1=0.0075
c1=0.03
#c1=3.187/1000
#c1=6.8325/100_000
Lb=7.14
pi=3.14
rho0=c0*6.023/10000 #units should be 1/A^3 for each rho quantity
rho0d=rho0*epsilon     #this is rho0'
rho1=c1*6.023/10000  #this is rho1+ for the electroneutral adsorbent (the same as rho1- in this case)
rhop1=c1*6.023/10000    #this is rho1+ for the new system
rhom1=rhop1+rho0d    #this is rho1- for the new system
a=4*pi*Lb*zp*z0
bp=zp*zp*8*pi*Lb*(rhop1)	#this is actually b+ squared
c=4*pi*rho0*Lb
alpha=a*zp/z0
beta=a*a*rho0
k0old=4*pi*Lb*(2*rho0*z0**2)/Q # k0 squared for the electroneutral adsorbent
k0=4*pi*Lb*(rho0*z0**2+rho0d*z0d**2)/Q #k0 squared for the new system
k1=zp*zp*4*pi*Lb*(2*rho1) #k1 squared for the electroneutral adsorbent


#old aka electroneutral adsorbent
q12_pp=-(Q*Lb*k0old**2/(k0old-k1)**2)*((np.exp(-np.sqrt(k0old)*r)/r)-(np.exp(-np.sqrt(k1)*r)/r)*(1-((r*np.sqrt(k1)/2)*(1-(k1/k0old)))))
q12_pm=-q12_pp


def q12n_pp(r,a,c,pi,Q,k0,w,zp,alpha,beta,bp):
    s=zp*zp*4*pi*Lb*(rhop1+rhom1*w*w)#this is actually s squared
    return (np.exp(-np.sqrt(s)*r)*(beta/(bp-s)-beta*c/(Q*(s-k0)*(s-bp)))+np.exp(-np.sqrt(k0)*r)*beta*c/(Q*(s-k0)*(k0-bp))-
         np.exp(-np.sqrt(bp)*r)*(beta/(bp-s)+beta*c/(Q*(k0-bp)*(s-bp))))/(4*pi*r)

def q12n_mm(r,a,c,pi,Q,k0,w,zp,alpha,beta,bp):
    s=zp*zp*4*pi*Lb*(rhop1+rhom1*w*w)#this is actually s squared
    sm=zp*zp*8*pi*Lb*rhom1*w*w #this is actually s- squared
    return (np.exp(-np.sqrt(s)*r)*(beta/(sm-s)-beta*c/(Q*(s-k0)*(s-sm)))+np.exp(-np.sqrt(k0)*r)*beta*c/(Q*(s-k0)*(k0-sm))-
         np.exp(-np.sqrt(sm)*r)*(beta/(sm-s)+beta*c/(Q*(k0-sm)*(s-sm))))*w*w/(4*pi*r)

def q12n_pm(r,a,c,pi,Q,k0,w,zp,alpha,beta,bp):
    s=zp*zp*4*pi*Lb*(rhop1+rhom1*w*w)#this is actually s squared
    sm=zp*zp*8*pi*Lb*rhom1*w*w #this is actually s- squared
    v1 = ((beta*(bp/2-sm)/w**2+alpha*beta*rhom1)*(bp/2-k0)*Q+beta*c*(alpha*rhom1+(bp/2-sm)/w**2))/((bp/2-sm)*(bp/2-s)*(bp/2-k0)*Q)
    v2 = ((beta*(sm-s)/w**2-alpha*beta*rhom1)*(s-k0)*Q+beta*c*(sm-s)/w**2-alpha*beta*rhom1*c)/((s-bp/2)*(sm-s)*(s-k0)*Q)
    v3 = ((alpha*beta*rhom1)*(sm-k0)*Q+alpha*beta*rhom1*c)/((sm-bp/2)*(sm-s)*(sm-k0)*Q)
    v4 = (beta*c*(sm-k0)/w**2-alpha*beta*rhom1*c)/((bp/2-k0)*(sm-k0)*(s-k0)*Q)
    return (np.exp(-np.sqrt(bp/2)*r)*(v1) + np.exp(-np.sqrt(s)*r)*v2 + np.exp(-np.sqrt(sm)*r)*v3 +np.exp(-np.sqrt(k0)*r)*v4)*w**3/(4*pi*r)


fig = plt.figure(figsize=(6,4))
ax=fig.add_subplot(111)
ax.set_xlim(0, rmax)
ax.set_ylim(-ymax, ymax)
f12n_pp,=ax.plot([],[],"g",linewidth=2,label="$q^{12}_{+\!\!+\!\!}$")
f12n_mm,=ax.plot([],[],"r",linewidth=2,label="$q^{12}_{-\!\!-\!\!}$")
f12n_pm,=ax.plot([],[],"b",linewidth=2,label="$q^{12}_{+\!\!-\!\!}$")
q_pp,=ax.plot(r,q12_pp,"r--",linewidth=2,label="$q^{12}_{+\!\!+\!\!}$")
q_pm,=ax.plot(r,q12_pm,"b--",linewidth=2,label="$q^{12}_{+\!\!-\!\!}$")
title=ax.set_title("GRAPH")
ax.set_xlabel("r / A",fontsize=14)
ax.set_ylabel("$q^{12}_{+\!\!+\!\!}$ , $q^{12}_{+\!\!-\!\!}$ , $q^{12}_{-\!\!-\!\!}$",fontsize=14)
ax.legend(loc="upper right",frameon=False,labelspacing=0.4,fontsize=10)
def animate(i):
    y=q12n_pp(r,a,c,pi,Q,k0,0.1*i+1,zp,alpha,beta,bp)
    z=q12n_mm(r,a,c,pi,Q,k0,0.1*i+1,zp,alpha,beta,bp)
    h=q12n_pm(r,a,c,pi,Q,k0,0.1*i+1,zp,alpha,beta,bp)
    title.set_text("razmerje nabojev -> w ="+" "+str(round(0.1*i+1,1)))
    f12n_pp.set_data(r,y)
    f12n_mm.set_data(r,z)
    f12n_pm.set_data(r,h)

ani = FuncAnimation(fig, func=animate, frames=91, interval=200,repeat=True)
#the upper line allows the user to adjust the speed and number of frames, which 
#in turn defines the upper limit of w (the bottom limit is set to 1)
fig.tight_layout()
ani.save("C:/Users/tibor mlakar/Desktop/slike magistrska/q12_ANIMACIJE/q12_2b.gif",fps=2)
#the upper line saves the animation at a specified location (note from my example)
