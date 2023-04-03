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


#Constantes 
rho = 0.8 #kg/L
T_in = 110 #°C
r = 1
D = 2*r
L = 5
kf = 317.33
kp = 0.00057

#Fonctions

def dvdh(L, r, h) :
    thetadvdh = [L, r]
    dvdh = (L*r**2)/(r*np.sqrt(1-((r-h)/r)**2)) + L*((-2*h**2+4*h*r-r**2)/np.sqrt(-h**2+2*h*r))
    return dvdh

def dhdt(L, r, h, rho, Fin, Fl, Fv) :
    thetadhdt = [L, r, rho]
    Fl = kf * np.sqrt(h)
    dhdt = (1/(rho*dvdh(L, r, h)) * (Fin - Fl - Fv))
    return dhdt


def dMdt():
    thetadMdt = []


def dpdt() :
    thetadpdt = []
    1/M 

def dTdt() : 
    thetadTdt = []




#Perturbation ( Question Q) ):
#Changer fonction par dhdt, etc...

def fonctionhauteur(t, x, theta, t_per):
    u = [Fin, Fv, Fl]
    
    if (t > t_per[0]):
        u = [Fin*0.02, Fv, Fl]
 
    return fun_init(x, theta, u)

def fonctionpression(t, x, theta, t_per):
    u = [Fin, Fv, Fl]
    
    if (t > t_per[0]):
        u = [Fin*0.02, Fv, Fl]
 
    return fun_init(x, theta, u)

def fonctiontempérature(t, x, theta, t_per):
    u = [Fin, Fv, Fl]
    
    if (t > t_per[0]):
        u = [Fin*0.02, Fv, Fl]
 
    return fun_init(x, theta, u)

def fonctionmasse(t, x, theta, t_per):
    u = [Fin, Fv, Fl]
    
    if (t > t_per[0]):
        u = [Fin*0.02, Fv, Fl]
 
    return fun_init(x, theta, u)



            
