""" The right-hand side part of the equation of motion, transformation of coordinates
"""

#This has been updated to use numba, which compiles the below function on the fly
#and substantially speeds up the ODE integration
from numba import cfunc, types
import numpy as np
import matplotlib.pyplot as plt

c_sig = types.void(types.double,
                   types.CPointer(types.double),
                   types.CPointer(types.double),
                   types.CPointer(types.double))

@cfunc(c_sig)
def funcpl(t,z,params,dzdt):
    """
    Takes the time and solution vector z, as well as parameters in the params array, 
    and populates the dzdt array with the values of the right-hand side function f(t,z).
    
    This function returns the right-hand side f(t,z) for a planet subject to graviational
    force due to the center body with GM_S = 4 pi^2
    
    The system:
    - the Sun
    - Mercury with an elliptic orbit
    - Venus, Earth, Mars, Jupiter, Saturn with circular orbits (radius = semi-major axis)
    
    Note: Venus is rotating in the opposite direction
    
    The order:
    0 Venus, 1 Earth, 2 Mars, 3 Jupiter, 4 Saturn
    
    params[i*3    ] = Mass of the planet in solar masses
    params[i*3 + 1] = Semi-major axis of the orbit in AU
    params[i*3 + 2] = Period of rotation in years
    
    Returns: nothing
    """
    xm  = z[0]
    Vmx = z[1]
    ym  = z[2]
    Vmy = z[3]
    
    rm = np.sqrt(xm**2 + ym**2)
    
    dzdt[0] = Vmx
    dzdt[1] = -4. * (np.pi ** 2) * xm / (rm ** 3)
    dzdt[2] = Vmy
    dzdt[3] = -4. * (np.pi ** 2) * ym / (rm ** 3)
    
    # compute coordinates of planets other than Mercury
    for plan in range(0,5):
        x_pl = params[plan*3 + 1] * np.cos(2. * np.pi * t / params[plan*3 + 2])
        y_pl = params[plan*3 + 1] * np.sin(2. * np.pi * t / params[plan*3 + 2])
        if plan == 0:
            y_pl *= -1

        r_mpl = np.sqrt( (xm - x_pl) ** 2 + (ym - y_pl) ** 2)
        
        dzdt[1] += - 4. * (np.pi ** 2) * params[plan*3] * (xm - x_pl) / (r_mpl ** 3)
        dzdt[3] += - 4. * (np.pi ** 2) * params[plan*3] * (ym - y_pl) / (r_mpl ** 3)
        
def ellipse_to_xy(a,e,theta,thetaE):
    """
    Takes the particle's position relative to an ellipse and parameters of the ellipse a,e,theta,theta_E.
    This function returns the Cartesian variables x,V_x,y,V_y.
    
    Returns x,Vx,y,Vy
    """

    # radius using angle theta
    r = a * (1 - e**2) / (1 + e * np.cos(theta - thetaE))
    
    # angular momentum per mass
    h = 2. * np.pi * np.sqrt(np.abs(a * (1. - e **2)))
    
    # energy per mass
    u = - 2. * (np.pi ** 2) / a 
    
    # speed of the particle
    V = np.sqrt(np.abs(2. * u + 8. * (np.pi ** 2) / r)) 
    
    # let Vx = V cos alpha, Vy = V sin alpha
    # buff = alpha - theta
    # when the radial velocity is positive (the planet goes from its periapse to apoapse = sin(theta-theta_E) > 0)
    # alpha - theta should be less then pi/2
    buff_sin = np.array(h / (r * V))
    
    # to make sure that arcsin takes values less than 1 and greater than -1
    buff_sin[buff_sin < -1.] = -1.
    buff_sin[buff_sin > 1.] = 1.
    
    buff = np.pi*(np.sin(theta - thetaE) < 0.) + np.power(-1., np.sin(theta - thetaE) < 0.) * np.arcsin(buff_sin)
    alpha = theta + buff
        
    # x and y
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    
    # Vx and Vy
    Vx = V * np.cos(alpha)
    Vy = V * np.sin(alpha)
    
    return x,Vx,y,Vy

def xy_to_ellipse(x,Vx,y,Vy):
    """
    Takes the Cartesian variables.
    This function returns the particle's position relative to an ellipse and parameters of the ellipse.
    
    Returns a,e,theta,theta_E
    """
    # radius using x and y
    r = np.sqrt(x ** 2 + y ** 2)
    
    # speed of the particle
    V = np.sqrt(Vx ** 2 + Vy ** 2)
    
    # angular momentum per mass
    h = x * Vy - y * Vx
    
    # energy per mass
    u = (V ** 2) / 2. - 4. * (np.pi ** 2) / r
    
    # semi-major axis
    a = -2. * ((np.pi) ** 2) / u
    
    # eccentricity of the elliptical orbit, added absolute value
    e = np.sqrt(np.abs(1 - ((h / (2. * np.pi)) ** 2 )/ a))
    
    # theta
    theta = np.arctan2(y,x)
    
    # theta_E, compute e*cos(theta - thetaE) first
    buff = a * (1. - e ** 2) / r - 1.
    
    # divide buff/e and output 0 if it is a circular orbit
    buff_cos = np.divide(buff, e, out=np.zeros_like(buff), where=(e > np.power(10.,-5.)))
    
    #to make sure that arccos takes values less than 1 and greater than -1
    buff_cos[buff_cos < -1.] = -1.
    buff_cos[buff_cos > 1.] = 1.
    
    delta = np.arccos(buff_cos)
    
    # change the sign if the radial velocity is negative
    delta *= np.power(-1.,(x * Vx + y * Vy) < 0.)
    thetaE = theta - delta
    
    # set thetaE to 0 if it is a circular orbit
    thetaE *= (e > np.power(10.,-5.))
    
    # fix to add 2pi or subtract 2pi if thetaE isn't between -pi and pi
    thetaE -= (thetaE > np.pi) * 2 * np.pi
    thetaE += (thetaE < -np.pi) * 2 * np.pi
    
    return a,e,theta,thetaE