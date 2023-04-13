import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


def Cp_ethanol(T):
    Cp=238.31+(-2.381)*(T+273.15)+(0.013317)*(T+273.15)**2-(3.2*10**-5)*(T+273.15)**3+(3.15*10**-8)*(T+273.15)**4
    return Cp

def Hvap(T):
    Hvap=44025-83.516*T+0.34032*T**2-0.002*T**3
    return Hvap

def antoine(T):
    Psat=10**(8.13484-(1662.48)/(T+238.131))
    Psat_pa=Psat*133.322
    return Psat_pa

def integrale_Cp(Tout,Tin):
    Tout+=273.15
    Tin+=273.15
    int_Cp = 238.31*(Tout-Tin) + -2.381*(Tout**2-Tin**2)/2 + 0.013317*(Tout**3-Tin**3)*1/3 -3.2*10**-5*((Tout**4)-(Tin**4))/4 + 3.15*10**-8*((Tout**5)-(Tin**5))/5
    return int_Cp

#paramètres
rho=0.8 #kg/L
Fin=57119.49036/3600/ rho * 0.001 #m^3/s
Fv=38079.66024/3600/rho * 0.001 #m^3/s
Fl=19039.83012/3600/rho * 0.001 # m^3/s
M_ethanol=46.07/1000
r=1
h =1
Wv_permanent = 38173/3600/rho * 0.001 #m^3/s
L=5
T=120
D=2
R = 8.314 

Tin=110
Tout= 120
kf= Fl
kp=Wv_permanent/(np.sqrt(r**2-(r-h)**2)*L*D*(antoine(T)-(3.3*101325)))
Q = Fv*Hvap(T)/M_ethanol + Fin*(Cp_ethanol(T) - integrale_Cp(120,25))/M_ethanol

Vtot = r**2*np.pi*L

print(r)
print(h)
def fonc_non_lin(x,theta,u):
    [h,T,m,P] = x
    [Fin,Fl,Fv] = u 
    [r,L,rho,R,M_ethanol,kp,Q]=theta 

    #print(h)
    hvap=Hvap(T)
    cp=Cp_ethanol(T)
    V = (np.arccos((r-h)/r)*r**2 - (np.sqrt(r**2 - (r-h)**2) * (r-h)))*L
    A = np.sqrt((r**2) - ((r-h)**2))*L*2 
    Wv = kp*(antoine(T)- 3.3*101325)*A
    dvdh = 10*np.sqrt(h*(2 - h)) #(L*r**2)/(r*np.sqrt(1-((r-h)/r)**2)) + L*((-2*h**2+4*h*r-r**2)/np.sqrt(-h**2+2*h*r))
    dhdt = (Fin- (kf*np.sqrt(h)) - Wv)/(rho* A)
    dTdt = (Fin*(integrale_Cp(Tin, 25)- (integrale_Cp(T, 25) * ( kf*np.sqrt(h) + Wv)))+Q*M_ethanol-Wv*hvap)/(rho*V*cp)
    dm_gazeuxdt = Wv-Fv
    dPdt = (P/T)*(dTdt)- (P/V)*(1/rho)*(dm_gazeuxdt) + (R*T*(Wv-Fv)/(M_ethanol*V))
    
    print(x)
    
    return [dhdt,dTdt,dm_gazeuxdt, dPdt]

def dxdt_non_lin(t,x,theta,t_per):
    u = [Fin, Fl, Fv]

    #Perturbation
    if (t> t_per[0]):
        u = [Fin, Fl,Fv]
    
    
    
    return fonc_non_lin(x,theta,u)






h_0 = 1
T_0 = 110 
V_0 = (np.arccos((r-h)/r)*r**2 - (np.sqrt(r**2 - (r-h)**2) * (r-h)))*L
m_0 = rho*V_0
P_0 = 3.3*101325 

theta = [r,L,rho,R,M_ethanol,kp,Q]
u_0 = [Fin,Fl,Fv]
x_0 = [h_0,T_0,m_0,P_0]

t_per = [2, 5]
tspan = [0, 20000]

sim_non_lin = solve_ivp(dxdt_non_lin,tspan,x_0, method='RK45', args=(theta, t_per), max_step=100)
p_non_lin = np.array([Fin if i > t_per[0]  else Fin for i in sim_non_lin.t])


plt.plot(sim_non_lin.t, p_non_lin,label='Fin')
plt.title("Figure X. Variable de perturbation Fin en fonction du temps")
plt.xlabel('Temps (s)')
plt.ylabel('Variation de Fin (kg/s)')
plt.legend()
plt.show()


plt.plot(sim_non_lin.t, sim_non_lin.y[0],label='h non linéaire')
plt.title("Évolution de h en fonction de t")
plt.xlabel('Temps (s)')
plt.ylabel('Hauteur h (m)')
plt.legend()
plt.show()

plt.plot(sim_non_lin.t, sim_non_lin.y[1],label='T non linéaire')
plt.title("Évolution de T en fonction de t")
plt.xlabel('Temps (s)')
plt.ylabel('Température (°C)')
plt.legend()
plt.show()

plt.plot(sim_non_lin.t, sim_non_lin.y[2],label='m non linéaire ')
plt.title("Évolution de m en fonction de t")
plt.xlabel('Temps (s)')
plt.ylabel('Masse (kg)')
plt.legend()
plt.show()

plt.plot(sim_non_lin.t, sim_non_lin.y[3],label='P non linéaire')
plt.title("Évolution de P en fonction de t")
plt.ylabel('Pression P (Pa) ')
plt.legend()
plt.show()
