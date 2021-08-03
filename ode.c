/*******************************************************************
 *
 *                    generic integrator
 *                    to compile:
 *                    gcc -shared -O2 ode.c -o libode.so
 *       
 *******************************************************************
*/
#include <stdio.h>
#include <stdlib.h>
#include <math.h>


/****** function prototypes begin ******/
int solve_ode(double *pt, double *px, double dt, int nsteps, int nvar, int order, double *params,
              void (*dxdt)(double t, double *px, double *params, double *derivs));
void stepEuler(double dt, double t, double a[], double anew[], int nvar,
               double params[], double *buf,
               void (*dxdt)(double t, double a[], double params[], double derivs[]));
void stepEulercromer(double dt, double t, double a[], double anew[], int nvar,
                     double params[], double *buf,
                     void (*dxdt)(double t, double a[], double params[], double derivs[]));;
void stepRK2(double dt, double t, double a[], double anew[], int nvar,
             double params[], double *buf,
             void (*dxdt)(double t, double a[], double params[], double derivs[])); 
void stepRK4(double dt, double t, double a[], double anew[], int nvar,
             double params[], double *buf,
             void (*dxdt)(double t, double a[], double params[], double derivs[])); 
void stepVelocityVerlet(double dt, double t, double a[], double anew[], int nvar,
    double params[], double* buf,
    void (*dxdt)(double t, double a[], double params[], double derivs[]));

void stepLeapFrog(double dt, double t, double a[], double anew[], int nvar,
    double params[], double* buf,
    void (*dxdt)(double t, double a[], double params[], double derivs[]));

void stepYoshida4(double dt, double t, double a[], double anew[], int nvar,
    double params[], double* buf,
    void (*dxdt)(double t, double a[], double params[], double derivs[]));
/****** function prototypes end ******/



/*****************************
*
* ODE library functions start 
*
******************************/

// Generic ODE solver:
// pt and px should hold nsteps+1 elements 
// nvar  = number of dependent variables 
// order = choose a method of integration:  1 = Euler, -1 = Euler-Cromer, 2 = RK2, 4 = RK4,
//                                          5 = Leap Frog, 6 = Yoshida4, 13 = Velocity Verlet (default)
int solve_ode(double *pt, double *px, double dt, int nsteps, int nvar, int order, double *params,
               void (*dxdt)(double t, double *px, double *params, double *derivs))
{
  double *buf;
  //pointer to time stepper function
  void (*step_ode)(double dt, double t, double a[], double anew[], int nvar,
                     double params[], double *buf,
                     void (*dxdt)(double t, double a[], double params[], double derivs[]));
  int step;

  if( nsteps < 0 ) {
    dxdt = NULL;
    nsteps *=- 1;
  }
  
  //allocate memory for storing temporary variables
  buf = calloc( 5*nvar, sizeof(double) );
  
  if(NULL == buf) {
    fprintf(stderr, "solve_ode(): could not allocate memory\n");
    return 1;
  }
  
  //choose solver
  switch(order){
    case 1:
      step_ode = &stepEuler;
      break;
    case -1: 
      if(2!=nvar) {
        fprintf(stderr, "Euler-Cromer only works for 2 variables\n");
        return -1;
      }
      step_ode = &stepEulercromer;
      break;
    case 2: 
      step_ode = &stepRK2;
      break;
    case 4:
      step_ode = &stepRK4;      
    case 5:
      step_ode = &stepLeapFrog;
      break;
    case 6:
      step_ode = &stepYoshida4;
      break;
    case 13:
    default:
        //use Velocity Verlet stepping by default
        step_ode = &stepVelocityVerlet;
  }
  
  if(NULL == dxdt) {
    //if no r.h.s. function provided, devault to damped forced pendulum
    dxdt = &func_pendulum;
  }
  
  for(step = 0; step < nsteps; step++) {
    pt[step+1] = pt[0] + (step+1)* dt;    
    /* take step */
    (*step_ode)(dt, pt[step], &px[step*nvar], &px[(step+1)*nvar], nvar, params, buf, dxdt); 
  }
  //free memory
  free(buf);
  buf = NULL;
  return 0;
}

/******** step  *********************/
/*** Take a step with the Euler method, output anew ***/
/*** Calls function derivs to get derivative ***/
void stepEuler(double dt, double t, double a[], double anew[], int nvar,
               double params[], double *buf,
               void (*dxdt)(double t, double a[], double params[], double derivs[]))
{
    int i ;
    double *dadt = buf ; 
    (*dxdt)(t, a, params, dadt) ; 
    for (i=0 ; i<nvar; i++) anew[i] = a[i] + dt*dadt[i] ; 
}

void stepEulercromer(double dt, double t, double a[], double anew[], int nvar,
                     double params[], double *buf,
                     void (*dxdt)(double t, double a[], double params[], double derivs[]))
{
    int i ;
    double *dadt = buf;
    (*dxdt)(t, a, params, dadt) ; 
    anew[1] = a[1] + dt*dadt[1] ; 
    anew[0] = a[0] + anew[1]*dt ; 
    
}

void stepRK2(double dt, double t, double a[], double anew[], int nvar,
             double params[], double *buf,
             void (*dxdt)(double t, double a[], double params[], double derivs[]))
{
    int i ;
    double *f1 = buf, *f2 = f1+nvar; 
    double *a2 = f2 + nvar; 
    (*dxdt)(t, a, params, f1); 
    for (i=0 ; i<nvar ; i++) a2[i] = a[i] + (0.5*dt)*f1[i]; 
    (*dxdt)(t+0.5*dt, a2, params, f2) ; 
    for (i=0 ; i<nvar ; i++) anew[i] = a[i] + dt*f2[i]; 
}

void stepRK4(double dt, double t, double a[], double anew[], int nvar,
             double params[], double *buf,
             void (*dxdt)(double t, double a[], double params[], double derivs[]))
{
    int i ;
    double *f1 = buf, *f2 = f1+nvar, *f3 = f2+nvar, *f4 = f3+nvar, *ax = f4+nvar; 
    const double sixth = 0.16666666666666666667; //save 1./6. to avoid recomputing division at every step
    
    (*dxdt)(t, a, params, f1); 
    for (i=0 ; i<nvar ; i++) ax[i] = a[i] + (0.5*dt)*f1[i] ; 
    (*dxdt)(t+0.5*dt, ax, params, f2); 
    for (i=0 ; i<nvar ; i++) ax[i] = a[i] + (0.5*dt)*f2[i] ; 
    (*dxdt)(t+0.5*dt, ax, params, f3); 
    for (i=0 ; i<nvar ; i++) ax[i] = a[i] + (dt)*f3[i] ; 
    (*dxdt)(t+dt, ax, params, f4); 
    
    for (i=0 ; i<nvar ; i++) 
      anew[i] = a[i] + sixth*(f1[i]+2.0*f2[i]+2.0*f3[i]+f4[i])*dt; 
}

/* source: https://young.physics.ucsc.edu/115/leapfrog.pdf*/
void stepLeapFrog(double dt, double t, double a[], double anew[], int nvar,
    double params[], double* buf,
    void (*dxdt)(double t, double a[], double params[], double derivs[]))
{
    int i;
    double* dadt = buf, * vhalf = buf + nvar;

    (*dxdt)(t, a, params, dadt);

    for (i = 0; i < nvar / 2; i++){
        vhalf[i] = a[2 * i + 1] + dadt[2 * i + 1] * 0.5 * dt;
    }
    
    for (i = 0; i < nvar / 2; i++){
        anew[2 * i] = a[2 * i] + vhalf[i] * dt;
    }

    (*dxdt)(t + dt, anew, params, dadt);

    for (i = 0; i < nvar / 2; i++){
        anew[2 * i + 1] = vhalf[i] + 0.5 * dadt[2 * i + 1] * dt;
    }
}

/* source: https://young.physics.ucsc.edu/115/leapfrog.pdf */
void stepVelocityVerlet(double dt, double t, double a[], double anew[], int nvar,
    double params[], double* buf,
    void (*dxdt)(double t, double a[], double params[], double derivs[]))
{
    int i;

    /* f1 and f2 saves the values of dVdt */
    /* halfV are velocities at 0.5*dt */
    double* f1 = buf, * f2 = f1 + nvar, * halfV = f2 + nvar;

    (*dxdt)(t, a, params, f1);

    for (i = 0; i < (nvar / 2); i++) {
        halfV[i] = *(a + 2 * i + 1) + 0.5 * f1[2 * i + 1] * dt;
    }

    /* compute new position */
    for (i = 0; i < (nvar / 2); i++) {
        *(anew + 2 * i) = *(a + 2 * i) + halfV[i] * dt;
    }

    (*dxdt)(t + dt, anew, params, f2);

    /* compute new velocity */
    for (i = 0; i < (nvar / 2); i++) {
        *(anew + 2 * i + 1) = halfV[i] + 0.5 * dt * f2[2 * i + 1];
    }
}

/* source: https://en.wikipedia.org/wiki/Leapfrog_integration#4th_order_Yoshida_integrator */
void stepYoshida4(double dt, double t, double a[], double anew[], int nvar,
    double params[], double* buf,
    void (*dxdt)(double t, double a[], double params[], double derivs[])) {
    int i, k;

    /* f stores the values of dVdt */
    /* temp_xV stores temporary values for x,V */
    /* coeff_xV includes 4 values for x and 3 values for V, order in x,V */

    double* f = buf, * temp_xV = f + nvar, * coeff_xV = temp_xV + nvar;

    /* set coefficients */
    const double num0 = -1.7024143839193153215916254;
    const double num1 = 1.3512071919596577718181152;

    /* c1, c2, c3, c4 coefficients for x */
    coeff_xV[0] = 0.5 * num1;
    coeff_xV[2] = 0.5 * (num0 + num1);
    coeff_xV[4] = coeff_xV[2];
    coeff_xV[6] = coeff_xV[0];

    /* d1, d2, d3 coefficients for V */
    coeff_xV[1] = num1;
    coeff_xV[3] = num0;
    coeff_xV[5] = coeff_xV[1];

    /* set temporary variables to initial conditions*/

    for (i = 0; i < nvar; i++) {
        temp_xV[i] = a[i];
    }

    /* take three Euler steps with different dt
    assume that dxdt = V and dVdt is defined by the given function */
    
    for (k = 0; k < 3; k++) {
        for (i = 0; i < (nvar / 2); i++) {
            temp_xV[2 * i] = temp_xV[2 * i] + coeff_xV[2 * k] * temp_xV[2 * i + 1] * dt;
        }

        (*dxdt)(t + coeff_xV[2 * k] * dt, temp_xV, params, f);

        for (i = 0; i < (nvar / 2); i++) {
            temp_xV[2 * i + 1] = temp_xV[2 * i + 1] + coeff_xV[2 * k + 1] * f[2 * i + 1] * dt;
        }
    }

    /* compute new values */
    for (i = 0; i < (nvar / 2); i++) {
        anew[2 * i] = temp_xV[2 * i] + coeff_xV[6] * temp_xV[2 * i + 1] * dt;
        anew[2 * i + 1] = temp_xV[2 * i + 1];
    }
}

/**************************
*
* ODE library functions end 
* 
**************************/
