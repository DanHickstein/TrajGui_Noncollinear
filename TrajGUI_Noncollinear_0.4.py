# DanHickstein@gmail.com

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, RadioButtons
from mpl_toolkits.mplot3d import Axes3D
import scipy.integrate

######################################################
##### Constants:
q  = 1.602e-19    #Coulombs   Charge of electron
c  = 3.0e8        #m/s        Speed of light
eo = 8.8541e-12   #C^2/(Nm^2) Permittivity of vacuum
me = 9.109e-31    #kg         Mass of electron
ke = 8.985551e9   #N m^2 C-2  Coulomb's constant
hbar = 1.05457173e-34 #m2 kg / s
######################################################


def plot_everthing(ax1,ax2,I1,I2,lambda1,lambda2,ellip1,ellip2,delay,
                   cross_angle,Tsteps,Tstart,Tduration):

    FWHM    = 2    # sec, just make the pulses really long
    fwhm1   = FWHM       
    fwhm2   = FWHM
    delay   = lambda1/c * delay  #delay between the two fields in sec
    theta1 = cross_angle
    theta2 = -cross_angle
    
    laser_params = I1,I2,lambda1,lambda2,fwhm1,fwhm2,ellip1,ellip2,delay,theta1,theta2

    t = np.linspace(Tstart * lambda1/c,(Tduration+Tstart)*lambda1/c,Tsteps)
    
    Ex,Ey,Ez = E_field(t,laser_params)
    Bx,By,Bz = E_field(t,laser_params,B_field=True)
    Bx = Bx*c; By=By*c; Bz=Bz*c
    
    ax1.plot(Ex,Ey,Ez,      color='r',label='E-Field',zorder=10)
    ax1.plot(Bx,By,Bz,color='b',label='B-Field',zorder=10)
    
    ax1.scatter(Ex[0],Ey[0],Ez[0],color='r',label='tb',zorder=10)
    ax1.scatter(Bx[0],By[0],Bz[0],color='b',label='tb',zorder=10)
    
    
    equal_axis(ax1,Bx,By,Bz)
    equal_axis(ax1,Ex,Ey,Ez)
        
    tb = lambda1/c * Tstart

    x,y,z,vx,vy,vz = integrate_trajectory(tb,laser_params,t)
    ax2.plot(x,y,z,color='g',label='With B',zorder=10)
    x,y,z,vx,vy,vz = integrate_trajectory(tb,laser_params,t,B=False)
    ax2.plot(x,y,z,color='m',label='Without B',zorder=10)
    equal_axis(ax2,x,y,z)
    
    ax2.scatter(x[0],y[0],z[0],color='b',label='Tunnel')
    ax2.scatter(0,0,0,color='r',label='Ion')
    
    for ax in (ax1,ax2):
        leg = ax.legend(scatterpoints = 1)
        leg.draw_frame(False)
    
    
    
def integrate_trajectory(tb,laser_params,times,return_info=False,coulomb=False,B=True):
    # Laser params = [I1,I2,lambda1,lambda2,fwhm1,fwhm2,ellip1,ellip2,delay]
    Ip = 15.1
    x0,y0,z0 = tunnel_position(tb,Ip,laser_params)
    vx0,vy0,vz0 = 0,0,0
    initial_conditions = [x0,y0,z0,vx0,vy0,vz0]

    params = [tb,laser_params]

    # Here is where the magic happens
    def diff_eq(variables,times2,params):
        x,y,z,vx,vy,vz = variables
        tb,laser_params = params
        r = np.sqrt(x**2+y**2+z**2)
        if coulomb == True:
            Ac = -ke*q**2/(r**2*me)
        else:
            Ac = 0
        Ex,Ey,Ez = E_field(times2,laser_params)
        Bx,By,Bz = E_field(times2,laser_params,B_field=True)
        
        if B == True:
            return [vx,vy,vz,
                 Ac*x/r - q/me * (Ex + By*vz - Bz*vy), #doing a manual cross product
                 Ac*y/r - q/me * (Ey + Bz*vx - Bx*vz),
                 Ac*z/r - q/me * (Ez + Bx*vy - By*vx)]
                
        else:
            return [vx,vy,vz,
                 Ac*x/r - q/me * (Ex ),
                 Ac*y/r - q/me * (Ey ),
                 Ac*z/r - q/me * (Ez )]

    solution,info = scipy.integrate.odeint(diff_eq,initial_conditions,times,args=(params,),
                     full_output=True,h0=1e-20,hmax=1e-15,mxstep=1000)

    x,y,z,vx,vy,vz = np.hsplit(solution,6)
    x = x[:,0]
    y = y[:,0]
    z = z[:,0]
    
    if return_info == False:
        return x,y,z,vx,vy,vz
    else:
        return x,y,z,vx,vy,vz,info

    
def equal_axis(ax,X,Y,Z):
    max_range = np.array([X.max()-X.min(), Y.max()-Y.min(), Z.max()-Z.min()]).max()
    Xb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][0].flatten() + 0.5*(X.max()+X.min())
    Yb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][1].flatten() + 0.5*(Y.max()+Y.min())
    Zb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][2].flatten() + 0.5*(Z.max()+Z.min())
    # Comment or uncomment following both lines to test the fake bounding box:
    for xb, yb, zb in zip(Xb, Yb, Zb):
       ax.plot([xb], [yb], [zb], 'w')


def E_field(t,laser_parameters,B_field=False):
    I1,I2,lambda1,lambda2,fwhm1,fwhm2,ellip1,ellip2,delay,theta1,theta2 = laser_parameters 
    w1  = c/lambda1 * 2. * np.pi #Angular frequency of the laser
    w2  = c/lambda2 * 2. * np.pi #Angular frequency of the laser
    Eo1 = np.sqrt(2*I1*10**4/(c*8.85e-12)) # Electric field in V/m
    Eo2 = np.sqrt(2*I2*10**4/(c*8.85e-12)) # Electric field in V/m
    
    #correct for ellipticity
    Eo1 = Eo1/np.sqrt(ellip1**2+1)
    Eo2 = Eo2/np.sqrt(ellip2**2+1)
    
    #make carrier and envelope
    Ex1 =        Eo1*np.sin(w1*t) *         gaussian(t,fwhm=fwhm1)
    Ey1 = ellip1*Eo1*np.cos(w1*t) *         gaussian(t,fwhm=fwhm1)
    Ex2 =        Eo2*np.sin(w2*(t+delay)) * gaussian(t,fwhm=fwhm2,avg=delay)
    Ey2 = ellip2*Eo2*np.cos(w2*(t+delay)) * gaussian(t,fwhm=fwhm2,avg=delay)
    
    if B_field == True:
        Ex1,Ey1,junk = cross_product((Ex1,Ey1,0),(0,0,1))
        Ex2,Ey2,junk = cross_product((Ex2,Ey2,0),(0,0,1))
        Ex1,Ey1,Ex2,Ey2 = Ex1/c,Ey1/c,Ex2/c,Ey2/c
    
    theta1 = theta1/180.*np.pi #convert degrees to rad
    theta2 = theta2/180.*np.pi
    
    E1x = Ex1 * np.cos(theta1)
    E1y = Ey1
    E1z = Ex1 * np.sin(theta1)
    
    E2x = Ex2 * np.cos(theta2)
    E2y = Ey2
    E2z = Ex2 * np.sin(theta2)
    
    Ex = E1x + E2x
    Ey = E1y + E2y
    Ez = E1z + E2z
    
    return Ex,Ey,Ez
    
def tunnel_position(tb,Ip,laser_parameters):
    I1,I2,lambda1,lambda2,fwhm1,fwhm2,ellip1,ellip2,delay,theta1,theta2 = laser_parameters
    Ip = Ip * q
    Ex,Ey,Ez = E_field(tb,laser_parameters)    
    E = np.sqrt(Ex**2+Ey**2+Ez**2)
    r = -Ip/(E*q)
    return r*Ex/E,r*Ey/E,r*Ez/E

def dot_product((x1,y1,z1),(x2,y2,z2)):
    return x1*x2 + y1*y2 + z1*z2
    
def cross_product((a1,a2,a3),(b1,b2,b3)):
    return a2*b3-a3*b2, a3*b1-a1*b3, a1*b2-a2*b1

def gaussian(x,avg=0,fwhm=1):
    return np.exp(-(x-avg)**2/(2*(fwhm / 2.35482)**2))
    
    

###################### Setting up the Figure #####################
fig = plt.figure(figsize=(12,10))
ax1 = plt.subplot(2,2,1,projection='3d')
ax2 = plt.subplot(2,2,3,projection='3d')

###Slider axes###
sl = 0.60
sw = 0.35
sh = 0.02
sv = 0.97 #start vertical
ss = -0.045 #spacing

axL1    = plt.axes([sl, sv+1*ss, sw, sh])
axL2    = plt.axes([sl, sv+2*ss, sw, sh])

axI1    = plt.axes([sl, sv+4*ss, sw, sh])
axI2    = plt.axes([sl, sv+5*ss, sw, sh])

axE1 = plt.axes([sl, sv+7*ss, sw, sh])
axE2 = plt.axes([sl, sv+8*ss, sw, sh])

axCross = plt.axes([sl, sv+10*ss, sw, sh])
axDelay = plt.axes([sl, sv+11*ss, sw, sh])

axtnum = plt.axes([sl, sv+14*ss, sw, sh])
axtsta = plt.axes([sl, sv+15*ss, sw, sh])
axtend = plt.axes([sl, sv+16*ss, sw, sh])


###Sliders###
sL1 = Slider(axL1, 'Wavelength 1 \n(nm)', 0,100000, valinit=10000)
sL2 = Slider(axL2, 'Wavelength 2 \n(nm)', 0,100000, valinit=10000)

sI1 = Slider(axI1, 'Intensity 1 \n(1e14 W/cm2)', 0,100, valinit=5)
sI2 = Slider(axI2, 'Intensity 2 \n(1e14 W/cm2)', 0,100, valinit=5)

sE1 = Slider(axE1, 'Ellipticity 1', -1,1, valinit=1)
sE2 = Slider(axE2, 'Ellipticity 2', -1,1, valinit=-1)

sCross = Slider(axCross, 'Crossing angle\n(degrees)', 0,180, valinit=4)
sDelay = Slider(axDelay, 'Phase-delay\n(W2 cycles)' ,-1,1, valinit=0.5)

stnum = Slider(axtnum, 'Num time steps', 0,2000, valinit=1000)
ststa = Slider(axtsta, 'Start time\n(W2 cycles)',    0,1, valinit=.0001)
stend = Slider(axtend, 'Time duration\n(W2 cycles)', 0,5, valinit=1)

plt.subplots_adjust(left=0.08,bottom=0.05,right=0.95,top=0.96,hspace=0.14)
###################### Done setting up the figure #####################


###################### This activates everytime something is clicked ###############
def update(val):
    # load values from sliders:
    L1 = sL1.val*1e-9
    L2 = sL2.val*1e-9

    I1 = sI1.val*1e14
    I2 = sI2.val*1e14
    
    E1 = sE1.val
    E2 = sE2.val
    
    cross_angle = sCross.val
    delay       = sDelay.val
    
    tnum = stnum.val 
    tsta = ststa.val 
    tend = stend.val 
    
    # Annoyingly, it's best to just clear and re-make the graphs
    ax1.clear(); ax2.clear()
    
    ax1.set_title('Electric and Magnetic Fields')
    ax1.set_xlabel('Ex (V/m)')
    ax1.set_ylabel('Ey (V/m)')
    ax1.set_zlabel('Ez (V/m)')
    
    ax2.set_title('Electron trajectories')
    ax2.set_xlabel('x (m)')
    ax2.set_ylabel('y (m)')
    ax2.set_zlabel('z (m)')
    
    for ax in (ax1,ax2): # Power limits are good to prevent lots of zeros.
        ax.get_xaxis().get_major_formatter().set_powerlimits((0, 0))
        ax.get_yaxis().get_major_formatter().set_powerlimits((0, 0))
        ax.w_zaxis.get_major_formatter().set_powerlimits((0, 0))
        

    # All of the magic happens in another function:
    plot_everthing(ax1,ax2,I1,I2,L1,L2,E1,E2,delay,cross_angle,tnum,tsta,tend)
     
    fig.canvas.draw_idle() # Update the plots.

    
for s in (sL1,sL2,sE1,sE2,sI1,sI2,sCross,sDelay,stnum,ststa,stend):
    s.on_changed(update)

update('init') #make the plots of the first time

plt.savefig('Trajectories.png',dpi=200)
plt.show()  