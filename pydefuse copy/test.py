"""
----------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------
Created on Mon Mar  2 09:27:19 2020
@author: H.R.A. ten Eikelder
program: OF-DFT radial atom solver for the TF equations.
Description: This program takes as input the TF equation and gives 
            as output the electron density of the material. In this case 
            the material is one atom. The atom will be simulated on the left 
            boundary and the 'infinite space' will be simulated on the right 
            boundary. 
----------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------
"""
from __future__ import print_function
import math
import numpy as np
from dolfin import *
import matplotlib as mpl
from matplotlib import pyplot as plt

plt.close('all')

"""-------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------
                    Plot solution and mesh
----------------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------------"""

def plotting_solve_result_normal(u,title):
    r = (mesh.coordinates())
    x = np.sqrt(r)
    y = [v*sqrt(u(v)) for v in r]
    
    #x = r
    #y = [u(v) for v in r]
    
    plt.figure()
    plt.title(title)
    plt.xlabel("Radial coordinate")
    plt.ylabel(title)
    plt.grid()
    #plt.plot(x,y)
    plot(u)
    # show the plots
    plt.show()
    return 


def plotting_solve_result(u,title):
    r = (mesh.coordinates())
    x = np.sqrt(r)
    y = [v*sqrt(u(v)) for v in r]
    
    #x = r
    #y = [u(v) for v in r]
    
    plt.figure()
    plt.title(title)
    plt.xlabel("Radial coordinate")
    plt.ylabel(title)
    plt.grid()
    plt.plot(x,y)
    #plot(u)
    # show the plots
    plt.show()
    return 
'''------------------------------------'''
class DensityFields(object):
    
    def __init__(self,n,gn2,lp):
        self.n = n
        self.gn2 = gn2
        self.lp = lp

class DensityWeakForm(object):
    
    def __init__(self,n,v):
        self.n = n
        self.v = v
        self.gradn2 = inner(grad(n),grad(n))
        self.lapln = div(grad(n))
        self.gradv = v.dx(0)
        
class DensityRadialWeakForm(object):
    
    def __init__(self,n,v):
        self.n = n
        self.v = v
        self.gradn = n.dx(0)
        self.gradv = v.dx(0)
        
class TF(object):
    
    CF=(3.0/10.0)*(3.0*math.pi**2)**(2.0/3.0)
    
    def energy_density_expr(self,densobj):
        return self.CF*pow(densobj.n,(5.0/3.0))
            
    def potential_expr(self,densobj):
        return (5.0/3.0)*self.CF*pow(densobj.n,(2.0/3.0))

    def potential_enhancement_expr(self,densobj):
        return 1

    def potential_weakform(self,densobj):
        return (5.0/3.0)*self.CF*densobj.n**(2.0/3.0)*densobj.v

    def energy_weakform(self,densobj):
        return self.CF*pow(densobj.n,(5.0/3.0))*densobj.v

    def energy_density_field(self,densobj, field):
        field.vector()[:] = self.CF*np.power(self.n.vector()[:],(5.0/3.0))
    
    def potential_field(self,densobj,field):
        field.vector()[:] = (5.0/3.0)*self.CF*np.power(self.n.vector()[:],(2.0/3.0))

func_tf = TF()

class Dirac(object):
    
    CX = 3.0/4.0*(3.0/math.pi)**(1.0/3.0)
    CF=(3.0/10.0)*(3.0*math.pi**2)**(2.0/3.0)
    
    def energy_density_expr(self,densobj):
        return -self.CX*pow(densobj.n,(4.0/3.0))
            
    def potential_expr(self,densobj):
        return (-4.0/3.0)*self.CX*pow(densobj.n,1.0/3.0)

    def potential_enhancement_expr(self,densobj):
        return (-4.0/5.0)*(self.CX/self.CF)*1.0/pow(densobj.n,(1.0/3.0))

    def potential_weakform(self,densobj):
        return (-4.0/3.0)*self.CX*pow(densobj.n,(1.0/3.0))*densobj.v

    def energy_weakform(self,densobj):
        return -self.CX*pow(densobj.n,(4.0/3.0))*densobj.v

    def energy_density_field(self,densobj, field):
        field.vector()[:] = -self.CX*np.power(self.n.vector()[:],(4.0/3.0))
    
    def potential_field(self,densobj,field):
        field.vector()[:] = (-4.0/3.0)*self.CX*np.power(self.n.vector()[:],(1.0/3.0))

func_dirac = Dirac()

class Weizsacker(object):
    
    CX = 3.0/4.0*(3.0/math.pi)**(1.0/3.0)
    CF=(3.0/10.0)*(3.0*math.pi**2)**(2.0/3.0)
    
    def energy_density_expr(self,densobj):
        return 1.0/8.0*densobj.gn2/densobj.n
            
    def potential_expr(self,densobj):
        return 1.0/8.0*densobj.gn2/pow(densobj.n,2.0) - 1.0/4.0*densobj.lp/densobj.n

    def potential_enhancement_expr(self,densobj):
        return 1.0/(8.0*self.CF)*densobj.gn2/pow(densobj.n,11.0/3.0) - 1.0/(4.0*self.CF)*densobj.lp/pow(densobj.n,8.0/3.0)

    def potential_weakform(self,densobj):
        return ((1.0/8.0*(dot(grad(densobj.n),grad(densobj.n))/(densobj.n*densobj.n))*densobj.v+(1.0/4.0*(dot(grad(densobj.n),grad(densobj.v)))/densobj.n)))

    def energy_weakform(self,densobj):
        return 1.0/8.0*dot(grad(densobj.n),grad(densobj.n))/densobj.n*densobj.v
    
    def potential_field(self,densobj,field):
        raise Exception("Not implemented.")

func_weizsacker = Weizsacker()

functionals = [TF(),\
               Dirac(),\
               Weizsacker()\
               ]      
"""-------------------------------------------------------------------------------------------
---------------------------------------------------------------------------------------------- 
                    Creating the mesh + Function Spaces + defining type of element
----------------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------------""" 
# Create mesh and define function space
start_x = 0.5
end_x = 6
amount_vertices = 200
mesh = IntervalMesh(amount_vertices,start_x, end_x) # Splits up the interval [0,1] in (n) elements 

print("MES COORDS",np.shape(mesh.coordinates()))

#Creation of Function Space
P1 = FiniteElement("P", mesh.ufl_cell(), 2)
element = MixedElement([P1,P1])

V = FunctionSpace(mesh, 'P', 2) # P stands for lagrangian elemnts, number stands for degree
W = FunctionSpace(mesh, element)

#Define radial coordinates based on mesh
r = SpatialCoordinate(mesh)[0] # r are the x coordinates. 

#Element H
#Z = Constant(1) # Hydrogen
#N = Z # Neutral

#Element Ne
#Z = Constant(10) # Neon
#N = Z # Neutral

#Element Kr
Z = Constant(36) # Krypton
N = Z # Neutral 


"""-------------------------------------------------------------------------------------------
---------------------------------------------------------------------------------------------- 
                    Creating and defining the boundary conditions
----------------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------------"""

#Defining the tolerance on the boundaries 
tol = 1E-14

#Defining the left boundary
def boundary_L(x, on_boundary):
    return on_boundary and near(x[0], start_x, tol)

#Defining the right boundary
def boundary_R(x, on_boundary):
    return on_boundary and near(x[0], end_x, tol)

#Defining expression on left boundary
n_L = Expression('0', degree=1)         
bc_L = DirichletBC(V, n_L, boundary_L)  
    
#Defining expression on right boundary
n_R = Expression('0', degree=1)
bc_R = DirichletBC(V, n_R, boundary_R) 
    
#collecting the left and right boundary in a list for the solver to read
bcs = [bc_L, bc_R]
"""-------------------------------------------------------------------------------------------
---------------------------------------------------------------------------------------------- 
                    Defning external potential v[r] and Initial density n_1[r]
----------------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------------"""
#---constants---
lamb = 1/9
C1 = 3/10*(3*math.pi**2)**(2/3)
C2 = 3/4*(3/math.pi)**(1/3)
C3 = lamb/8
mu = 1
omega = Constant(1)

A_ = 4*C3
B_ = (-5/3)*C1
D_ = 1#(4/3)*C2

A__ = (5/3)*(C1/C3)
B__ = 4/3
C__ =  1/C3

#----- External Potential ----

#External potential v_e[r] is analytically described for atoms 
Ex = -Z/r

#------ Initial density ------

n_i = Expression('a*exp(-b*pow(x[0], 2))', degree = 2, a = 1, b=1.0)
#n_i = Expression('pow(x[0],2)', degree = 2)
#n_i = Constant(1)


nlast = Function(V)

#------------Checking amount of electrons ----------------------
u_n = interpolate(n_i,V)
intn = float(assemble((u_n)*dx(mesh)))
print("Number of electrons before adjustment:"+str(intn))
u_n.vector()[:] = u_n.vector()*N/intn  
intn = float(assemble((u_n)*dx(mesh)))
print("Number of electrons after adjustment:",intn)

"""-------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------
                    Defining and solving the variational problem
                    defining trial and test functions
----------------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------------"""
v_h = interpolate(Constant(-1), V)

mixed_test_functions = TestFunction(W)
(qr, pr) = split(mixed_test_functions)

du_trial = TrialFunction(W)        
du = Function(W)

# Put the initial density and v_h in u_k

u_k = Function(W)
assign(u_k.sub(0), v_h)
assign(u_k.sub(1), u_n)


bcs_du = []

eps = 1
iters = 0
maxiter = 200
minimal_error = 1E-9

while eps > minimal_error and iters < maxiter:
    iters += 1 
    
    (v_hk, u_nk) = split(u_k)
    
    #---- Setting up functionals -------------------
    densobj = DensityRadialWeakForm(u_nk, pr)
    funcpots = 0
    for f in functionals:
        if isinstance(f,tuple):
            funcpots += Constant(f[0]*f[1].potential_weakform(densobj))
        else:
            funcpots += f.potential_weakform(densobj)
            
    #---- Solving v_h and u_n ----------------------
   
    ## Variational equation With Radial transformation 
    #--- Rewriting variables for overview ---
    U_ = sqrt(u_nk)
    E_ = (-Ex - v_hk - mu)
    
    F = - A_*U_.dx(0)*qr.dx(0)*dx      \
        + 2*A_*(1/r)*U_.dx(0)*qr*dx \
        + B_*u_nk**(7/6)*qr*dx      \
        + D_*u_nk**(5/6)*qr*dx      \
        - Ex*U_*qr*dx              \
        - v_hk*U_*qr*dx            \
        + mu*U_*qr*dx
    
    #Poisson equation of Hatree potential
    F = F -(                          \
        - v_hk.dx(0)*pr.dx(0)*dx    \
        + (2/r)*v_hk.dx(0)*pr*dx  \
        + 4*math.pi*u_nk*pr*dx)
    
    
    #Calculate Jacobian
    J = derivative(F, u_k, du_trial)
    
    #Assemble system
    A, b = assemble_system(J, -F, bcs_du)

    #nvec = u_k.sub(1).vector()
    rvec = mesh.coordinates()
    nvec = np.array([v_hk(v) for v in rvec])
    minval = nvec.min()
    print("v_hk minimum:",minval)

    rvec = mesh.coordinates()
    nvec = np.array([u_nk(v) for v in rvec])
    minval = nvec.min()
    print("u_nk minimum:",minval)    
    
    #plotting_solve_result(v_hk,"v_nk pre solver")
    plotting_solve_result_normal(u_nk,"u_nk pre solver")
    
    solve(A, du.vector(), b)

    # Conserve memory by reusing vectors for v_h and u_n to keep dv_h and du_n
    dv_h = v_h                  #step to transfer type info to dv_h
    assign(dv_h, du.sub(0))     #step to transfer values to dv_h
    v_h = None                  #empty v_h
    
    du_n = u_n                  #step to transfer type info to du_n
    assign(du_n, du.sub(1))     #step to transfer values to du_n
    u_n = None                  #empty u_n

    #plotting_solve_result(dv_h,"dv_h")
    plotting_solve_result(du_n,"du_n") 
    
    #---- Calculate the Error -----------------------
    avg = sum(du.vector().get_local())/len(du.vector().get_local())
    eps = np.linalg.norm(du.vector().get_local()-avg, ord=np.Inf)
    if math.isnan(eps):
            raise Exception("Residual error is NaN")

    print("EPS is:",eps)
        
     #---- Taking the step for u_n and v_h ------------
    
    u_k.vector()[:] = u_k.vector()[:] + omega*du.vector()[:]
        
    # Conserve memory by reusing vectors for u_n, v_h also as du_n, dv_h
    v_h = dv_h 
    assign(v_h, u_k.sub(0))
    dv_h = None
    
    u_n = du_n 
    assign(u_n, u_k.sub(1))
    du_n = None  

    plotting_solve_result(u_n,"density")
    
    #---- Ad hoc negative density fix -------

    omega = 1 
    nvec = u_n.vector()
    minval = nvec.min()
    print("Going to add:",minval+0.5)
    if minval <= 0.0:
        nvec[:]=nvec[:]-minval+0.5
        intn = float(assemble((u_n)*dx(mesh)))
        print("Number of electrons before correction:",intn)
        nvec[:] = nvec[:]*N/intn    
        intn = float(assemble((u_n)*dx(mesh)))            
        assign(u_k.sub(1),u_n)
        print("Number of electrons after correction:",intn)
        

    nvec = u_k.sub(1).vector()
    minval = nvec.min()
    print("Minval after correction:",minval)
    
    print('Check end of loop, iteration: ', iters,' Error: ', eps)
