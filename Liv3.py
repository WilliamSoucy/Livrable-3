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
def dvdh(thetadvdh, h) :
    [L, r] = thetadvdh
    dvdh = (L*r**2)/(r*np.sqrt(1-((r-h)/r)**2)) + L*((-2*h**2+4*h*r-r**2)/np.sqrt(-h**2+2*h*r))
    return dvdh

thetadhdt = [L, r, rho]
def dhdt(L, r, h, rho, Fin, Fl, Fv) :
    [L, r, rho] = thetadhdt
    Fl = kf * np.sqrt(h)
    dhdt = (1/(rho*dvdh(L, r, h)) * (Fin - Fl - Fv))
    return dhdt


def dmdt():
    thetadmdt = []


def dpdt() :
    thetadpdt = []
    1/M 

def dTdt() : 
    thetadTdt = []




#Perturbation ( Question Q) ):
#t_per = temps perturbation

def dhdt_nonlin(t, x, thetadhdt, t_per):
    u = [Fin, Fv, Fl]
    
    if (t > t_per[0]):
        u = [Fin*0.02, Fv, Fl]
 
    return dhdt(x, thetadhdt, u)


def dpdt_nonlin(t, x, thetadpdt, t_per):
    u = [Fin, Fv, Fl]
    
    if (t > t_per[0]):
        u = [Fin*0.02, Fv, Fl]
 
    return dpdt(x, thetadpdt, u)


def dTdt_nonlin(t, x, thetadTdt, t_per):
    u = [Fin, Fv, Fl]
    
    if (t > t_per[0]):
        u = [Fin*0.02, Fv, Fl]
 
    return dTdt(x, thetadTdt, u)


def dmdt_nonlin(t, x, thetadmdt, t_per):
    u = [Fin, Fv, Fl]
    
    if (t > t_per[0]):
        u = [Fin*0.02, Fv, Fl]
 
    return dmdt(x, thetadmdt, u)

##################################################################################

#Simulation du système ODE non linéarisé

dhdtsim_nlin = solve_ivp(dhdt_nonlin,tspan,0.5, method='RK45', args=(thetadhdt, t_per), max_step=0.01)
