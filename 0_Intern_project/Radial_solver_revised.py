"""
----------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------
Created on Mon 22/04/2020 09:45

@author: H.R.A. ten Eikelder
program: OF-DFT radial atom solver for DFT equations. TFDW functionals included
Description: This program takes as input the OFDFT equation and gives 
            as output the electron density of the material. In this case 
            the material is one atom. The nucleus will be simulated on the left 
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

class DensityRadialWeakForm(object):
    
    def __init__(self,n,v):
        self.n = n
        self.v = v
        self.gradn = n.dx(0)
        self.gradv = v.dx(0)
        
class Mesh:
    def __init__(self):
        # Create mesh and define function space
        self.start_x = 0.1
        self.end_x = 20
        self.amount_vertices = 200
        self.mesh = IntervalMesh(amount_vertices,start_x, end_x) # Splits up the interval [0,1] in (n) elements 
        
        #Define radial coordinates based on mesh
        self.r = SpatialCoordinate(mesh)[0] # r are the x coordinates. 

    def custom_mesh():
        return True 
       
class Spaces:
    def __init__(self):
        #Creation of Function Space
        P1 = FiniteElement("P", mesh.ufl_cell(), 2) # defining mesh elements
        element = MixedElement([P1,P1]) # Mixed element for creation mixed function space
        
        self.V = FunctionSpace(mesh, 'P', 2) # P stands for lagrangian elemnts, number stands for degree
        self.W = FunctionSpace(mesh, element) # Created mixed function space
    
    def custom_spaces():
        return True
    
class Functions:
    def __init__(self):
        self.n = Function(V)
        self.n_i = function(V)
        self.nlast = Function(V)
        
        self.mixed_test_functions = TestFunction(W)
        (self.qr, self.pr) = split(mixed_test_functions)
        
        self.u_k = Function(W)
        self.du_trial = TrialFunction(W)        
        self.du = Function(W)
        

class Element:
    def __init__(self):
        self.Z = Constant(36) # Krypton
        
    def element(element):
       # element = library with elements and corersponding electrical charge
       return True
   
class Boundaries:
    
    #Defining the left boundary
    def boundary_L(x, on_boundary):
        return on_boundary and near(x[0], self.start_x, tol)

    #Defining the right boundary
    def boundary_R(x, on_boundary):
        return on_boundary and near(x[0], self.end_x, tol)
        
    def __init__(self):
        #Defining the tolerance on the boundaries 
        self.tol = 1E-14
        
        #Defining expression on left boundary
        n_L = Expression('0', degree=1)         
        bc_L = DirichletBC(V, n_L, boundary_L)  
        
        #Defining expression on right boundary
        n_R = Expression('0', degree=1)
        bc_R = DirichletBC(V, n_R, boundary_R) 
        
        #collecting the left and right boundary in a list for the solver to read
        bcs = [bc_L, bc_R]
        bcs_du = []
        
    def custom_boundaries():
            return True
        
class Initial_Density:
    def __init__(self):
        self.n = Expression('a*exp(-b*pow(x[0], 2))', degree = 2, a = 1, b=0.1)
    
    def start_dens(self, n, uniform, exp):
        if uniform == True and exp == True:
            print("invalid entry, choose 1 starting density")
            
        elif uniform == True:
            self.n = Constant(1)
            self.u_n = Interpolate(n,V)
            
        elif exp == True:
            self.n = Expression('a*exp(-b*pow(x[0], 2))', degree = 2, a = 1, b=0.1)
        else :
            print("initial density is uniform")
    
    def Initial_electron_adjustment(n):
        intn = float(assemble((n)*dx(mesh)))
        print("Number of electrons before adjustment:"+str(intn))
        n.vector()[:] = n.vector()*N/intn  
        intn = float(assemble((n)*dx(mesh)))
        print("Number of electrons after adjustment:",intn)

class Initial_potential:            
    def __init__(self, Z):
        Ex = -Z/r
        
           
class Negative_density_fix:
    def __init__(self):
        u_k = self.u_k
        du = self.du
        n = self.n
        
    def ad_hoc(n):
        nvec = n.vector()
        minval = nvec.min()
        
        if minval <= 0.0:
            nvec[:]=nvec[:]-minval+1
            
        intn = float(assemble((n)*dx(mesh)))
        print("Number of electrons before correction:",intn)
        
        nvec[:] = nvec[:]*N/intn    
        intn = float(assemble((n)*dx(mesh)))     
        assign(u_k.sub(1),n)
        print("Number of electrons after correction:",intn)
        
class Error_calculation:
    def __init__(self):
        dv_h = self.dv_h
        du = self.du
        dn = self.dn
        
    def calculate_error():
        avg = sum(dv_h.vector().get_local())/len(dv_h.vector().get_local())
        eps = np.linalg.norm(du.vector().get_local()-avg, ord=np.Inf)
        if math.isnan(eps):
                raise Exception("Residual error is NaN")

class Conserve_memory:
    def __init__():
        dv_h = self.dv_h
        u_k = self.u_k
        du_n = self.du_n
        
        v_h = self.v_h
        du = self.du
        u_n = self.u_n
        
    def n_to_dn(dv_h,u_k,du_n):
        # Conserve memory by reusing vectors for u_n, v_h also as du_n, dv_h
        v_h = dv_h                  #step to transfer type info to v_h
        assign(v_h, u_k.sub(0))     #step to transfer values to v_h
        dv_h = None                 #empty dv_h
    
        u_n = du_n                  #step to transfer type info to u_n
        assign(u_n, u_k.sub(1))     #step to transfer values to u_n
        du_n = None                 #empty du_n
    
    def dn_to_n(v_h,du,u_n):
        # Conserve memory by reusing vectors for v_h and u_n to keep dv_h and du_n
        dv_h = v_h                  #step to transfer type info to dv_h
        assign(dv_h, du.sub(0))     #step to transfer values to dv_h
        v_h = None                  #empty v_h
    
        du_n = u_n                  #step to transfer type info to du_n
        assign(du_n, du.sub(1))     #step to transfer values to du_n
        u_n = None   
        
class solve:
    def solve_basic_TF():
        return True
    def solve_vh():
       """
       - calculate functionals
       - ?do something with mu?
       - density calculation step 1
       - vh step
       - density calculation step 2
       - interpolate between steps
       - calculate the gradient
       """
    def solve_n():
        """
       - calculate functionals
       - do something with mu
       - density calculation step 1
       - vh step
       - density calculation step 2
       - interpolate between steps
       - calculate the gradient
       """
    def calculate_energies():
        """
        here will come the energy calcualtions after finishing the radial solver 
        based on visual confirmation with the results of the paper  
        'Accurate solution of the Thomas-Fermi-Dirac-Weizsacker variational 
        equations for the case of neutral atoms and positive ions'
        """