int solve_ode(double *pt, double *px, double dt, int nsteps, int nvar, int order, double *params,
void (*dxdt)(double t, double *px, double *params, double *derivs));
