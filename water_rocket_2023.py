#!/usr/bin/env python
# coding: utf-8

# # Water Rocket
# 
# As you saw in class a water rocket is an example of a system that accelerates losing mass. The center of mass will move at straight constant motion:
# $$\frac{dP_{ext}}{dt}=F_{ext}$$
# This equation sets the condition for the rocket equation:
# $$M\frac{dv}{dt}-u\frac{dM}{dt}=F_{ext}$$
# Where $M$ is the total mass of the rocket (varying in time), $v$ is the speed of the rocket and $u$ is the speed of the fuel relative to the rocket. Note that here we will define $u$ as positive pointing upward so that it matches the sign of $v$. The second term on the left side is also called thrust.
# 
# The external forces on the rocket in our experiment are gravity and air resistance, making $F_{ext}=-Mg-\beta v^2$ for drag coefficient $\beta$.
# 
# In the case of the water rocket, the speed of the exhaust (the water), won't be constant, so the equation will be solved numerically. Fist of all, we need to find a relation between the exhaust's speed ($u$) and the mass of the water left in the bottle. The water is pushed out of the bottle at high speed by the air at pressure ($P$). Bernoulli equation gives us the dynamical properties of the water:
# $$P_1+\frac{1}{2}\rho u_1^2+\rho gh_1=P_2+\frac{1}{2}\rho u_2^2+\rho gh_2$$
# With $\rho$ and $u$ are the density and the speed of the fluid, while the index 1,2 correspond to the value of the physical quantity measured in two different points. At the air/water interface inside the bottle $P$ with a height $h$ and the water level is decreasing at speed $u_{in}$. At the nozzle the pressure will be $P_{atm}$ with speed $u$.
# $$P+\frac{1}{2}\rho u_{in}^2+\rho gh=P_{atm}+\frac{1}{2}\rho u^2$$
# By conservation of mass: $u_{in}A_b=uA_n$ where $A_b$ and $A_n$ are the cross sectional area of the bottle and the nozzle respectively. Substituting this relation into the previous, we can estimate the speed of the exhausted fuel (we consider the negative solution, since the fuel will be expelled downwards):
# $$u=-\sqrt{\frac{2(P-P_{atm}+\rho gh)}{\rho\big(1-\frac{A_n^2}{A_b^2}\big)}}$$
# The terms $\rho gh$ and $\frac{A_n^2}{A_b^2}$ will be much smaller than the terms they are added to (think about why), so we drop them and plug this relation into the rocket equation:
# $$M\frac{dv}{dt}=-\sqrt{\frac{2}{\rho}(P-P_{atm})}\frac{dM}{dt}-gM-\beta v^2$$
# The expression we just found has two unknown quantities, both varying in time ($M,P$) and only one equation. We can use the ideal gas model to estimate the pressure of the expanding gas as function of its volume, hence as function of the mass of the water ($m$). If we assume the expansion is adiabatic (is this a good assumption? You could also try an isothermal expansion instead), then, in formulas:
# $$P_0V_0^{\gamma}=PV^{\gamma}$$
# With $\gamma=\frac{7}{5}$ for a diatomic gas and $P_0$, $V_0$ are the initial pressure and volume of the air respectively.
# $$P=P_0V_0^{\gamma}V^{-\gamma}=P_0V_0^{\gamma}\Big(V_0+\frac{m_0-m}{\rho}\Big)^{-\gamma}=P_0\Big(1+\frac{m_0-m}{V_0\rho}\Big)^{-\gamma}$$
# $m_0$ is the initial mass of the water. This yields:
# $$M\frac{dv}{dt}=-\sqrt{\frac{2}{\rho}(P_0\Big(1+\frac{m_0-m}{V_0\rho}\Big)^{-\gamma}-P_{atm})}\frac{dM}{dt}-gM-\beta v^2$$
# As last we observe that the rate of mass loss is proportional to $u$:$\frac{dM}{dt}=\frac{dm}{dt}=\rho\frac{dV}{dt}=\rho A_b\frac{dz}{dt}$, $\frac{dz}{dt}$ is the speed at which the level is decreasing in the bottle $u_{in}$, so using again the conservation of mass: $\frac{dM}{dt}=\rho A_{n}u$ 

# In[2]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np


# In[196]:


###all the quantities are measured in MKS
g        = 9.81                                # gravitational acceleration in m/s^2
r_nozzle = 0.0045                             #radius of nozzle in m
r_bottle = 0.0525                             #radius of bottle in m
rho      = 1000.0                             # water density in kg/m^3
p_meas   = 50                                 #this is measured gauge pressure in psi
p_pas    = 6895                               #conversion factor from psi to pascal
p_atm    = 101325
p_0      = p_meas*p_pas + p_atm              #initial pressure in the bottle - gauge pressure + atmospheric pressure, converted to pascal
gamma    = 1.4
air_density = 1.2
beta_o   = 0.5*0.0102561085*0.494*air_density                             #simulated drag coefficient
beta_c   = 0.5*0.0214212475*1.407*air_density 
beta_f   = 0.5*0.0102561085*1.018*air_density 
beta_s   = 0.5*0.0102561085*0.536*air_density                          #this is the drag coefficient of a cone but make your own estimate                            #average air density at sea level in kg/m^3                                                       #initial water mass in kg
beta     = beta_o
m_0      = 0.5
m_rocket = 0.045+0.0815+0.03                                                   #mass of the empty rocket
m_o      = 0.096
m_c      = 0.137
m_f      = 0.077
m_s      = 0.056                                                     #initial water mass in kg
m_bottle = m_rocket + m_o                                                       #mass of the empty rocket
V_bottle = 2/1000.0                                                 #volume of the bottle in m^3
V_0      = V_bottle - m_0/rho                                         #initial volume of the air

exhaust_speed = lambda m: np.sqrt(2/rho*(p_0*np.power(1+(m_0-m)/(V_0*rho),-gamma)-p_atm))     #this is u
mass_loss     = lambda m: np.pi*np.square(r_nozzle)*rho*exhaust_speed(m)                      #this is dm/dt

if np.power(p_0/p_atm,1.0/gamma)*V_0 < V_bottle:
    raise 'expansion factor too low, fuel is wasted, please increase/decrease initial pressure/mass'


# By integrating the equation:
# $$\frac{dm}{dt}=\rho S_{n}u=-\rho S_{n}\sqrt{\frac{2}{\rho}(P_0\Big(1+\frac{m_0-m}{V_0\rho}\Big)^{-\gamma}-P_{atm})}$$
# We can estimate how long the propulsion lasts and the mass decreases over time

# In[197]:


t = np.linspace(0,0.4,1000)  #creates a vector of timesteps from 0 to 0.3 s in 1000 steps. Defines your integration step size and total integration time
mass = np.zeros(len(t))
m = m_0
i = 0
while m>0:
    try:
        mass[i] = m
        m -= mass_loss(m)*t[1]
        i += 1
        
    except IndexError: break

t    = t[mass>0]
mass = mass[mass>0]
print('length of propulsion: %.2fms'%(t[-1]*1000))
t2 = np.linspace(t[-1],2.5,20000)


# At this point we can numerically solve the rocket equation and analyze the kinematics:
# $$\frac{dv}{dt}=\frac{u}{M}\frac{dM}{dt}-g-\beta\frac{v^2}{M}$$

# In[198]:


speed = np.zeros(len(t))
sp = 0
for i in range(len(t)):
    sp += exhaust_speed(mass[i])/(mass[i]+m_bottle)*mass_loss(mass[i])*t[1]-g*t[1]-beta*np.square(sp)/(mass[i]+m_bottle)*t[1]
    speed[i] = sp
    
distance = np.cumsum(speed*t[1])
fig, ax1 = plt.subplots()
ax1.set_ylabel('distance [m]', color='red')
ax1.set_xlabel('time [ms]')
ax1.scatter(t*1000,distance,color='red')
ax2 = ax1.twinx()
ax2.set_ylabel('speed [m/s]', color='blue')
ax2.scatter(t*1000,speed,color='blue')
print( 'speed at the end of propulsion:%.1fm/s'%speed[-1])
print( 'height at end of propulsion:%.1fm'%distance[-1])


# In[ ]:





# This gives you the height and speed of the rocket when it runs out of fuel. After this, the rocket will keep rising but the physics will just be that of projectile motion. You'll need to repeat the above process to calculate the total height of the rocket. Once the propulsion phase is over the rocket will have a mass $M_e$ and velocity $v_{p}$, the dynamics now is simpler:
# $$\frac{dv}{dt}=-g-\frac{\beta}{M_e}v^2$$
# 
# Use the space below to copy the code above and modify it so you can find the maximum height. Make sure you also start with the velocity that the first phase ended at. It'll also be helpful to define new arrays for time, speed, etc. because you'll need to integrate for much longer in time. E.g. define a new t2 that will integrate over the time your rocket was in the air, and then make corresponding speed2 and distance2 vectors from it.

# In[199]:


speed2 = np.zeros(len(t2))
sp = speed[-1]
for i in range(len(t2)):
    sp += -g*(t2[1]-t2[0])-beta*np.square(sp)/(m_bottle)*(t2[1]-t2[0])
    speed2[i] = sp
    
distance2 = np.cumsum(speed2*(t2[1]-t2[0]))
fig, ax1 = plt.subplots()
ax1.set_ylabel('distance [m]', color='red')
ax1.set_xlabel('time [ms]')
ax1.scatter(t2*1000,distance2,color='red')
ax2 = ax1.twinx()
ax2.set_ylabel('speed [m/s]', color='blue')
ax2.scatter(t2*1000,speed2,color='blue')
print( 'speed at the end of trajectory:%.1fm/s'%speed2[-1])
print('max height:%.1fm'%max(distance2))


# In[200]:


MVe = m_bottle*g*distance[-1]+0.5*m_bottle*speed[-1]**2
MHe = m_bottle*g*max(distance2)+0.5*m_bottle*speed2[np.argmax(distance2)]**2
print('max velocity energy:%.1fJ'%MVe)
print('max height energy:%.1fJ'%MHe)
print('energy loss:%.1fJ'%(MVe-MHe))


# In[213]:


barWidth = 0.3

br1 = np.arange(4)
br2 = [x + barWidth for x in br1] 
br3 = [x + barWidth for x in br2] 

plt.bar(br1,[27.5,25.9,22.6,18.4], color = 'tomato', width = barWidth, label = 'Simulated')
plt.bar(br2,[19.90,17.25,25.66,16.17],yerr=[5.24,3.64,12.97,5.14], color = '#52a49a', 
        width = barWidth, label = 'Experimental')


plt.xticks([r + 0.5*barWidth for r in range(4)], 
        ['Concave','Flat','Sphere','Ogive'])

plt.xlabel("Nose Cone Shape")
plt.ylabel("Average Energy Loss (J)")
plt.title("Energy Loss for Various Nose Cones")
plt.legend(fontsize = 10, loc = 'upper right')
plt.show()
plt.savefig('barchart.pdf')


# In[ ]:




