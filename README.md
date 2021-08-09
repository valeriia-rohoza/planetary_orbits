# planetary_orbits
Modeling the planetary orbits and predicting the stability of the systems (C, Python)

## Description of the model
This is the model of the Solar System. It includes Mercury, Venus, Earth, Mars, Jupiter, Saturn, and the Sun. We assume that all planets but Mercury rotate around the Sun at the circular orbits with the known parameters, and we set the initial conditions of Mercury. Then we integrate equations of motion of Mercury using the symplectic (area-preserving) ordinary differential equation (ODE) solver, written in C. To run the simulation and analyze data, see the Jupyter Notebook ```ode.ipynb``` or ```ode.pdf``` for pdf version of the Jupyter Notebook.  

## Integrating the Mercury orbit
We can choose one of five methods to numerically solve the equations of motion of Mercury. ```Yoshida4``` does a good job since it is of the 4th order and it conserves the total energy of the system.

Feel free to play around with the time step ```dt```, the duration of the simulation ```time_f```, and the numerical methods (Euler ```"Euler"```, Euler-Cromer  ```"Euler-Cromer"```, Runge-Kutta 2nd order ```"RK2"```, Runge-Kutta ```"RK4"```, Velocity Verlet ```"Velocity Verlet"```, Leapfrog ```"Leapfrog"```, Yoshida 4th oder ```"Yoshida4"```.

We can plot the orbit of Mercury using the ```Matplotlib``` library.
<a href="https://imgur.com/bzgbD9v"><img src="https://i.imgur.com/bzgbD9v.png?1" title="source: imgur.com" /></a>

## Determining the stability of the system
To determine whether the system is stable, we compute the ```stability``` parameter, which corresponds to 1 if Mercury is still within 10 astronomical units (AU) and to 0 if not.

## Running simulations with pertubations of the Mercury's orbit
We can randomize the initial conditions of Mercury and explore different setups of the planetary orbits. In this example, we set different values of the semi-major axis ```am``` and the eccentricity ```em```. That's how we obtain the training and the testing data sets for the neural network model. 

## Neural Network
To predict the stability of the system, we implement neural network with one hidden layer (see ```model.py```). Based on the testing data set, the accuracy of the prediction is 91.6%

## Feedback and comments
would be really helpful!

## Credits
To call C functions inside Jupyter Notebook, I utilized Professor Sasha Tchekhovskoy's approach.
