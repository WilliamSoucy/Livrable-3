import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve

########################

def Cp_ethanol(T):
    Cp=231+(-2.381)*T+(0.0133317)*T**2-(3.2*10**-5)*T**3+(3.15*10**-8)*T**4
    return Cp

def Hvap(T):
    Hvap=44025-83.516*T+0.34032*T**2-0.002*T**3
    return Hvap

def antoine(T):
    Psat=10**(8.13484-(1662,48)/(T+238.131))
    return Psat

##############################################################

#Paramètre
rho = 0.8 #kg/L
T_in = 110 #°C
r = 1 #m
D = 2*r #m
L = 5 #m
kf = 317.33
kp = 0.00057 

Fin = 57119 / 3600 #kg/s

h = 0.5 #initialement...

Fl = kf * np.sqrt(h)

Fv = Fin-Fl

# Temps de simulation
t_per = [1]
tspan = [0, 20]


#############################################################

#Fonctions

thetadvdh = [L,r]
def dvdh(x,thetadvdh, u) :
    [v]=x
    [L, r] = thetadvdh
    dvdh = (L*r**2)/(r*np.sqrt(1-((r-h)/r)**2)) + L*((-2*h**2+4*h*r-r**2)/np.sqrt(-h**2+2*h*r))
    return dvdh

thetadhdt = [L, r, rho]
def dhdt(x, thetadhdt, u) :
    [h] = x
    [L, r, rho] = thetadhdt
    Fl = kf * np.sqrt(h)
    dhdt = (1/(rho*dvdh(x,thetadvdh, u)) * (Fin - Fl - Fv))
    return dhdt

thetadmdt = []
def dmdt():
    [] = thetadmdt

thetadpdt = []
def dpdt() :
    [] = thetadpdt

thetadTdt = []
def dTdt() : 
    [] = thetadTdt




#Perturbation ( Question Q) ):
#t_per = temps perturbation

def dhdt_nonlin(t, x, thetadhdt, t_per):
    u = [Fin, Fv, Fl]
    
    if (t > t_per[0]):
        u = [Fin*0.02, Fv, Fl]
 
    return dhdt(x, thetadhdt, u)


'''def dpdt_nonlin(t, x, thetadpdt, t_per):
    u = [Fin, Fv, Fl]
    
    if (t > t_per[0]):
        u = [Fin*0.02, Fv, Fl]
 
    return dpdt(x, thetadpdt, u)'''


"""def dTdt_nonlin(t, x, thetadTdt, t_per):
    u = [Fin, Fv, Fl]
    
    if (t > t_per[0]):
        u = [Fin*0.02, Fv, Fl]
 
    return dTdt(x, thetadTdt, u)"""


'''def dmdt_nonlin(t, x, thetadmdt, t_per):
    u = [Fin, Fv, Fl]
    
    if (t > t_per[0]):
        u = [Fin*0.02, Fv, Fl]
 
    return dmdt(x, thetadmdt, u)'''

##################################################################################

#Simulation du système ODE non linéarisé

dhdtsim_nlin = solve_ivp(dhdt_nonlin,tspan,[0.5], method='RK45', args=(thetadhdt, t_per), max_step=0.01)

#dpdtsim_nlin = solve_ivp(dpdt_nonlin,tspan,[0.5], method='RK45', args=(thetadpdt, t_per), max_step=0.01)

#dTdtsim_nlin = solve_ivp(dTdt_nonlin,tspan,[0.5], method='RK45', args=(thetadTdt, t_per), max_step=0.01)

#dmdtsim_nlin = solve_ivp(dmdt_nonlin,tspan,[0.5], method='RK45', args=(thetadmdt, t_per), max_step=0.01)


################################################################################

#Graphiques 

#Variable de perturbation :
ut_dhdt = np.array([Fin if i < t_per[0] else Fin*1.02 for i in dhdtsim_nlin.t])

plt.plot(dhdtsim_nlin.t, ut_dhdt, 'b-',linewidth=3)
plt.xlabel('Temps (min)')
plt.ylabel('Fin')
plt.title('Variables de perturbation Fin en fonction du temps')
plt.legend(['Fin'])
plt.show()

#Variable hauteur :
plt.plot(dhdtsim_nlin.t, dhdtsim_nlin.y[0], 'b-',linewidth=3)
plt.xlabel('Temps (min)')
plt.ylabel('h (m)')
plt.title("Variable d'état h en fonction du temps ")
plt.legend(['h (m)'])
plt.show()

#Variable pression :
plt.plot(dpdtsim_nlin.t, dhdtsim_nlin.y[0], 'b-',linewidth=3)
plt.xlabel('Temps (min)')
plt.ylabel('pression (pa)')
plt.title("Variable d'état p en fonction du temps ")
plt.legend(['p (pa)'])
plt.show()

#Variable température :
plt.plot(dTdtsim_nlin.t, dhdtsim_nlin.y[0], 'b-',linewidth=3)
plt.xlabel('Temps (min)')
plt.ylabel('Température (C)')
plt.title("Variable d'état T en fonction du temps ")
plt.legend(['T (C)'])
plt.show()

#Variable masse :
plt.plot(dmdtsim_nlin.t, dhdtsim_nlin.y[0], 'b-',linewidth=3)
plt.xlabel('Temps (min)')
plt.ylabel('masse (kg)')
plt.title("Variable d'état m en fonction du temps ")
plt.legend(['m (kg)'])
plt.show()
