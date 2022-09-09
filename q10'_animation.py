#program that animates q10' correlation functions derived in:
#screening of ionic interactions in a charged adsorbent - 
#the application of Replica Ornstein - Zernike equations 
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
zp=1 #this is the charge number of cations - annealed fluid
z0=1 #this is the charge number of cations - matrix
z0d=1 #this is the charge number of anions - template
rmax=30 #this is the radius or the size of the x axis
r=np.arange(0.1,rmax,0.1)
ymax=0.4 #this is the value of q10' functions or the size of the y axis
Q=0.4
epsilon=1
c0=0.425
#c1=0.005
#c1=0.03
c1=3.187/1000
#c1=6.8325/100_000
Lb=7.14
pi=3.14
rho0=c0*6.023/10000 #units should be 1/A^3 for each rho quantity
rho0d=rho0*epsilon     #this is rho0'
rho1=c1*6.023/10000     #this is rho1+ for the electroneutral adsorbent (the same as rho1- in this case)
rhop1=c1*6.023/10000    #this is rho1+ for the new system
rhom1=rhop1+rho0d    #this is rho1- for the new system
a=4*pi*Lb*zp*z0
c=4*pi*rho0*Lb
k0old=4*pi*Lb*(2*rho0*z0**2)/Q # k0 squared for the electroneutral adsorbent
k0=4*pi*Lb*(rho0*z0**2+rho0d*z0d**2)/Q #k0 squared for the new system
k1=zp*zp*4*pi*Lb*(2*rho1) #k1 squared for the electroneutral adsorbent


def q10_p0d(r,a,c,pi,Q,k0,w,zp):
    s=zp*zp*4*pi*Lb*(rhop1+rhom1*w*w)
    return -(a*c/(4*pi*r*Q*(k0-s)))*(-np.exp(-np.sqrt(k0)*r)
             +np.exp(-np.sqrt(s)*r))
def q10_m0d(r,a,c,pi,Q,k0,w,zp):
    s=zp*zp*4*pi*Lb*(rhop1+rhom1*w*w)
    return w*(a*c/(4*pi*r*Q*(k0-s)))*(-np.exp(-np.sqrt(k0)*r)
             +np.exp(-np.sqrt(s)*r))

fig = plt.figure(figsize=(6,4))
ax=fig.add_subplot(111)
ax.set_xlim(0, rmax)
ax.set_ylim(-ymax, ymax)
q_p0dot,=ax.plot([],[],"r",linewidth=2,label="$q^{10'}_{+\,\,\!\!0'\!\!}$")
q_m0dot,=ax.plot([],[],"b",linewidth=2,label="$q^{10'}_{-\,\,\!\!0'\!\!}$")
title=ax.set_title("GRAPH")
ax.set_xlabel("r / A",fontsize=14)
ax.set_ylabel("$q^{10'}_{+\,\,\!\!0'\!\!}$ , $q^{10'}_{-\,\,\!\!0'\!\!}$",fontsize=14)
labels=["label1","label2"]
ax.legend(loc="upper right",frameon=False,labelspacing=0.4,fontsize=12)
def animate(i):
    r=np.arange(0.1,rmax,0.1)
    y=q10_p0d(r,a,c,pi,Q,k0,0.1*i+1,zp)
    z=q10_m0d(r,a,c,pi,Q,k0,0.1*i+1,zp)
    title.set_text("razmerje nabojev -> w ="+" "+str(round(0.1*i+1,1)))
    q_p0dot.set_data(r,y)
    q_m0dot.set_data(r,z)

ani = FuncAnimation(fig, func=animate, frames=91, interval=200,repeat=True)
#the upper line allows the user to adjust the speed and number of frames, which 
#in turn defines the upper limit of w (the bottom limit is set to 1)
fig.tight_layout()
ani.save("C://Users/tibor mlakar/Desktop/slike magistrska/q10'_ANIMACIJE/q10'_2a_Q=0,4.gif",fps=2)
#the upper line saves the animation at a specified location (note from my example)
