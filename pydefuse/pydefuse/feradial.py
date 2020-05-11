# -*- coding: utf-8 -*-
# 
#    This file is part of PyDeFuSE
#    Copyright (C) 2015 Rickard Armiento
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU Affero General Public License as
#    published by the Free Software Foundation, either version 3 of the
#    License, or (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU Affero General Public License for more details.
#
#    You should have received a copy of the GNU Affero General Public License
#    along with this program.  If not, see <http://www.gnu.org/licenses/>.

from dolfin import *
#from mshr import *
import numpy
import pydefuse.gui
import pylab

# This is necessary because dolfin overrides object creation so that you cannot
# override __init__ to take your own arguments

class FeRadialLattice(object):
    
    def __init__(self, rs, params):        
        self.rs = rs
        self.num_vertices = len(rs)
        self.grid_start = rs[0]
        self.grid_end = rs[-1]
        self.vol = 4.0/3.0*pi*(rs[-1]**3-rs[0]**3)
        self.msh = None
        self.V = None
        self.params = params

        #parameters['cache_dir'] = '/tmp/fcc'

        parameters['allow_extrapolation'] = True
        #parameters["form_compiler"]["log_level"]    = INFO
        #parameters["form_compiler"]["cpp_optimize"] = True
        #parameters["form_compiler"]["cpp_optimize_flags"] = "-O3"
        parameters["form_compiler"]["optimize"]     = True
        parameters["linear_algebra_backend"] = "PETSc"
        if self.params['threadpar']:
            parameters["num_threads"] = 6

    def create_mesh(self,Z):
        self.msh = IntervalMesh(self.num_vertices-1,self.grid_start,self.grid_end)
        self.msh.coordinates()[:] = numpy.transpose(numpy.array([self.rs]))
        self.V = FunctionSpace(self.msh, "CG", 1)
            
    def refine_mesh(self, field, field_grad, field_lapl):
        raise Exception("Not implemented")
        
    def transfer_scalar_field(self, field):
        return interpolate(field,self.V)

    def new_scalar_field(self):
        return Function(self.V)

    def expression(self,expr,**vals):
        vals['element'] = self.V.ufl_element()
        return Expression(expr, **vals)

    def constant(self,value):
        return Constant(value)

    def scalar_field_from_expression(self,expression):
        return interpolate(expression,self.V)

    def scalar_field_from_expr(self,expr,**vals):
        return interpolate(self.expression(expr,**vals),self.V)

    def new_vector_field(self):
        VV = VectorFunctionSpace(self.fe.msh, 'CG', 1)
        return Function(VV)

    def integrate_scalar_field(self,field):
        r = SpatialCoordinate(self.msh)[0]
        return 4.0*pi*float(assemble((field*r*r)*dx(self.msh)))

    def grad_lapl(self,field):
        r = SpatialCoordinate(self.msh)[0]
        grad_field = project(field.dx(0),self.V)
        grad_field2 = project(field.dx(0)**2,self.V)
        lapl_field = project(grad_field.dx(0) + (2/r)*grad_field,self.V)            
        return grad_field2, lapl_field

    # The laplacian appear to get worse oscillations this way
    def grad_lapl_old(self,field):
        bcs = []
        
        grad_field2 = project(field.dx(0)**2,self.V)
        lapl_field = Function(self.V)
        
        r = SpatialCoordinate(self.msh)[0]
        u = TrialFunction(self.V)
        v = TestFunction(self.V)        
        a = u*v*dx
        # Works but with oscillations
        #L = (-1.0/(4.0*pi))*(2/r*vh.dx(0)*v-vh.dx(0)*v.dx(0))*dx
        L = (2/r*field.dx(0)*v-field.dx(0)*v.dx(0))*dx

        #a = inner(vh,v)*dx
        #L = (-1.0/(4.0*pi))*inner(u.dx(0),v.dx(0))*dx+2/r*inner(u.dx(0),v)*dx
        solve(a == L, lapl_field,bcs)

        
        
        #r = SpatialCoordinate(self.msh)[0]
        #grad_field = project(field.dx(0),self.V)
        #grad_field2 = project(field.dx(0)**2,self.V)
        #lapl_field = project(grad_field.dx(0) + (2/r)*grad_field,self.V)            
        return grad_field2, lapl_field

    def vh_to_dens(self,vh,dens):
        bcs = []
        
        r = SpatialCoordinate(self.msh)[0]
        u = TrialFunction(self.V)
        v = TestFunction(self.V)        
        a = u*v*dx
        # Works but with oscillations
        #L = (-1.0/(4.0*pi))*(2/r*vh.dx(0)*v-vh.dx(0)*v.dx(0))*dx
        L = (-1.0/(4.0*pi))*(2/r*vh.dx(0)*v-vh.dx(0)*v.dx(0))*dx

        #a = inner(vh,v)*dx
        #L = (-1.0/(4.0*pi))*inner(u.dx(0),v.dx(0))*dx+2/r*inner(u.dx(0),v)*dx
        solve(a == L, dens,bcs)

    def radial_outer_boundary_factory(self):
        end = self.grid_end    
        class OuterBoundary(SubDomain):        
            def inside(self, x, on_boundary):
                return bool(((near(x[0], end))))       
        return OuterBoundary()

def solve_scaled_radial_tf(fel, sol, Z):
    """
        Solutions come out that are n(x) rather than n(r) with x = alpha*r. This means the solution is exactly the same
        for any atom, but, the grid is less practical.
    """

    V = fel.V
    msh = fel.msh
    vol = fel.vol

    class InnerBoundary(SubDomain):        
        def inside(self, x, on_boundary):
            return bool(((near(x[0], fel.rs[0]))))       
    
    # Create Dirichlet boundary condition
    dbc0 = InnerBoundary()
    bc0 = DirichletBC(V, Constant(1.0), dbc0)

    dbc1 = fel.radial_outer_boundary_factory()
    bc1 = DirichletBC(V, Constant(0.0), dbc1)
    
    # Collect boundary conditions
    bcs = [bc0,bc1]

    # Define variational problem
    u = TrialFunction(V)
    v = TestFunction(V)
    x = SpatialCoordinate(msh)[0]
    F  = -u.dx(0)*v.dx(0)*dx - (1.0/(sqrt(x)))*(sqrt(u)**3)*v*dx
    u_ = Function(V)     # the most recently computed solution
    F  = action(F, u_)
    J = derivative(F, u_, u)

    problem = NonlinearVariationalProblem(F, u_, bcs, J)
    solver  = NonlinearVariationalSolver(problem)
    prm = solver.parameters
    prm['newton_solver']['absolute_tolerance'] = 1E-6
    prm['newton_solver']['relative_tolerance'] = 1E-6
    prm['newton_solver']['maximum_iterations'] = 100
    prm['newton_solver']['relaxation_parameter'] = 1.0
    if False:
        prm['newton_solver']['linear_solver'] = 'gmres'
        prm['newton_solver']['preconditioner'] = 'ilu'
        prm['newton_solver']['krylov_solver']['absolute_tolerance'] = 1E-9
        prm['newton_solver']['krylov_solver']['relative_tolerance'] = 1E-7
        prm['newton_solver']['krylov_solver']['maximum_iterations'] = 1000
        prm['newton_solver']['krylov_solver']['gmres']['restart'] = 40
        prm['newton_solver']['krylov_solver']['preconditioner']['ilu']['fill_level'] = 0
    set_log_level(PROGRESS)

    solver.solve()
     
    gr = project(u_.dx(0),V)
    alpha = 4.0/1.0*(2.0*Z/(9.0*pi**2))**(1.0/3.0)
    nx = project(alpha**3*Z/(4.0*pi)*gr.dx(0)/x,V)    

    sol.assign(nx)
    #N = 4.0*pi*float(assemble(nx*x/alpha*x/alpha*1.0/alpha*dx(msh)))
    #print "CHECK N(x)",N

def fake_solve_radial_tf(fel, sol, Z):
    rs = fel.rs
    alpha = 4.0/1.0*(2.0*Z/(9.0*pi**2))**(1.0/3.0)

    c1,c2,c3,c4 = 1.4712, 0.4973, 0.3875, 0.002102
    phir = numpy.power(1.0+c1*alpha*rs-c2*numpy.power(alpha*rs,3.0/2.0)+c3*numpy.power(alpha*rs,2.0)+c4*numpy.power(alpha*rs,3.0),-1.0)
    # Wolfram alpha output for second derivative:
    # (2 (a c1 - (3 a c2 Sqrt[x])/2 + 2 a c3 x + 3 a c4 x^2)^2)/(1 + a c1 x - a c2 x^(3/2) + a c3 x^2 + a c4 x^3)^3 -
    # (2 a c3 - (3 a c2)/(4 Sqrt[x]) + 6 a c4 x)/(1 + a c1 x - a c2 x^(3/2) + a c3 x^2 + a c4 x^3)^2
    phirpp = (2.0*numpy.power(alpha*c1-3.0/2.0*alpha*c2*numpy.sqrt(rs)+2.0*alpha*c3*rs+3.0*alpha*c4*numpy.power(rs,2.0),2.0))/\
            numpy.power(alpha*c1*rs-alpha*c2*numpy.power(rs,3.0/2.0)+alpha*c3*numpy.power(rs,2.0)+alpha*c4*numpy.power(rs,3.0)+1.0,3.0) -\
            (-(3.0*alpha*c2)/(4.0*numpy.sqrt(rs))+2.0*alpha*c3+6.0*alpha*c4*rs)/\
            numpy.power(alpha*c1*rs-alpha*c2*numpy.power(rs,3.0/2.0)+alpha*c3*numpy.power(rs,2.0)+alpha*c4*numpy.power(rs,3.0)+1.0,2.0)
    dens = Z/(4.0*pi)*numpy.divide(phirpp,rs)
    
    raise Exception("Not implemented yet.")
    #TODO: Rewrite as dolfin expression for sol instead, or just correctly match gridpoints...

def solve_radial_tf(fel, sol, Z, screening_function=None):

    # Note, we have sign-changed both left and right hand sides
    V = fel.V
    msh = fel.msh
    vol = fel.vol

    alpha = 4.0/1.0*(2.0*Z/(9.0*pi**2))**(1.0/3.0)

    class InnerBoundary(SubDomain):        
        def inside(self, x, on_boundary):
            return bool(((near(x[0], fel.rs[0]))))       
    
    # Create Dirichlet boundary condition
    dbc0 = InnerBoundary()
    bc0 = DirichletBC(V, Constant(1.0), dbc0)

    dbc1 = fel.radial_outer_boundary_factory()
    bc1 = DirichletBC(V, Constant(0.0), dbc1)
    
    # Collect boundary conditions
    bcs = [bc0,bc1]

    # Define variational problem
    u = TrialFunction(V)
    v = TestFunction(V)
    r = SpatialCoordinate(msh)[0]
    F  = -u.dx(0)*v.dx(0)*dx - alpha**(3.0/2.0)*(1.0/(sqrt(r)))*(sqrt(u)**3)*v*dx
    u_ = Function(V)     # the most recently computed solution
    F  = action(F, u_)
    J = derivative(F, u_, u)

    problem = NonlinearVariationalProblem(F, u_, bcs, J)
    solver  = NonlinearVariationalSolver(problem)
    prm = solver.parameters
    prm['newton_solver']['absolute_tolerance'] = 1E-6
    prm['newton_solver']['relative_tolerance'] = 1E-6
    prm['newton_solver']['maximum_iterations'] = 100
    prm['newton_solver']['relaxation_parameter'] = 1.0
    if False:
        prm['newton_solver']['linear_solver'] = 'gmres'
        prm['newton_solver']['preconditioner'] = 'ilu'
        prm['newton_solver']['krylov_solver']['absolute_tolerance'] = 1E-9
        prm['newton_solver']['krylov_solver']['relative_tolerance'] = 1E-7
        prm['newton_solver']['krylov_solver']['maximum_iterations'] = 1000
        prm['newton_solver']['krylov_solver']['gmres']['restart'] = 40
        prm['newton_solver']['krylov_solver']['preconditioner']['ilu']['fill_level'] = 0
    set_log_level(PROGRESS)

    solver.solve()
     
    gr = project(u_.dx(0),V)
    nr = project(Z/(4.0*pi)*gr.dx(0)/r,V)

    sol.assign(nr)
    if screening_function != None:
        screening_function.assign(u_)
        
    #rsqrn = project(alpha*self.Z/(4.0*pi)*(gr.dx(0)*(x)),V)
    #rsqrn.vector()[:] = numpy.sqrt(rsqrn.vector())    
    
    #N = 4.0*pi*float(assemble(sol*r*r*dx(msh)))
    #print "CHECK N(r)",N

def solve_radial_tf_method2(fel, sol, Z, screening_function=None):

    V = fel.V
    msh = fel.msh
    vol = fel.vol

    alpha = 4.0/1.0*(2.0*Z/(9.0*pi**2))**(1.0/3.0)
    
    def inner_boundary(x, on_boundary):
        return on_boundary and near(x[0],fel.rs[0])
    
    def outer_boundary(x, on_boundary):
        return on_boundary and near(x[0],fel.rs[-1])
    
    Gamma_0 = DirichletBC(V, Constant(0.0), outer_boundary)
    Gamma_1 = DirichletBC(V, Constant(1.0), inner_boundary)
    bcs = [Gamma_0, Gamma_1]

    u_i = interpolate(Constant(0), V)  # previous (known) u
    u = TrialFunction(V)
    v = TestFunction(V)
    r = SpatialCoordinate(msh)[0]
    a = -u.dx(0)*v.dx(0)*dx 
    L = alpha**(3.0/2.0)*(1.0/(sqrt(r)))*(sqrt(u_i)**3)*v*dx
    A, b = assemble_system(a, L, bcs)
    u_k = Function(V)
    solve(A, u_k.vector(), b)

    Gamma_0_du = DirichletBC(V, Constant(0), outer_boundary)
    Gamma_1_du = DirichletBC(V, Constant(0), inner_boundary)
    bcs_du = [Gamma_0_du, Gamma_1_du]

    du_ = TrialFunction(V)
    F  = -u_k.dx(0)*v.dx(0)*dx - alpha**(3.0/2.0)*(1.0/(sqrt(r)))*(sqrt(u_k)**3)*v*dx
    J = derivative(F, u_k, du_)

    du = Function(V)
    u  = Function(V)  # u = u_k + omega*du
    omega = 1.0       # relaxation parameter
    eps = 1.0
    tol = 1.0E-5
    iter = 0
    maxiter = 25
    while eps > tol and iter < maxiter:
        iter += 1
        A, b = assemble_system(J, -F, bcs_du)
        solve(A, du.vector(), b)
        eps = numpy.linalg.norm(du.vector().array(), ord=numpy.Inf)
        print ('Norm:', eps)
        u.vector()[:] = u_k.vector() + omega*du.vector()
        u_k.assign(u)
     
    gr = project(u_k.dx(0),V)
    nr = project(Z/(4.0*pi)*gr.dx(0)/r,V)

    sol.assign(nr)
    if screening_function != None:
        screening_function.assign(u_k)
        
    #rsqrn = project(alpha*self.Z/(4.0*pi)*(gr.dx(0)*(x)),V)
    #rsqrn.vector()[:] = numpy.sqrt(rsqrn.vector())    
    
    #N = 4.0*pi*float(assemble(sol*r*r*dx(msh)))
    #print "CHECK N(r)",N

        
def solve_radial_poisson_old(fel, sol, rhs, Z = 0, boundaryexpr=None):

    # Note, we have sign-changed both left and right hand sides
    V = fel.V
    msh = fel.msh
    vol = fel.vol
    
    r = SpatialCoordinate(msh)[0]
    #N = 4.0*pi*float(assemble(rhs*r*r*dx(msh)))
    #print "CHECK N",N

    rhs.vector()[:] = 4.0*pi*(rhs.vector())
    
    # Create Dirichlet boundary condition
    dbc = fel.radial_outer_boundary_factory()         
    if boundaryexpr == None:
        boundaryexpr = Constant(0.0)
    bc0 = DirichletBC(V, boundaryexpr, dbc)
    
    # Collect boundary conditions
    bcs = [bc0]

    # Define variational problem
    u = TrialFunction(V)
    v = TestFunction(V)
    #r = V.cell().x[0]
    #a = u.dx(0)*v.dx(0)*dx - (2/r)*u.dx(0)*v*dx
    #a = u.dx(0)*v.dx(0)*r*r*dx - 2*r*u.dx(0)*v*dx
    a = 4.0*pi*u.dx(0)*v.dx(0)*r*r*dx 
    L = 4.0*pi*rhs*r*r*v*dx 
    #L = rhs*v*dx 

    A,b = assemble_system(a, L, bcs)
    if Z != 0:
        print ("APPLYING DELTA",Z)
        # TODO: Should we have delta^3? How do we do that?...
        delta = PointSource(V, Point(0.0),4.0*pi*Z)
        delta.apply(b)
    
    print("= Solving radial poisson equation")
    solve(A, sol.vector(), b)
    print ("= Finished")
    #plot(sol,interactive=True)
     
    #drawgrid = numpy.r_[0:10:0.1]
    #drawgrid = numpy.logspace(5,6,100)
    #drawgrid -= drawgrid[0]
    #gui.plot_radial(drawgrid,rhs)
    #gui.pause()
    #oneoverr = fel.scalar_field_from_expr("Z/x[0]",Z=Z)
    #gui.plot_radial(drawgrid,[sol,oneoverr])
    #gui.plot_radial(drawgrid,sol)
    #gui.pause()
    #exit(0)



def solve_radial_poisson(fel, sol, rhs, Z = 0, boundaryexpr=None):

    # Note, we have sign-changed both left and right hand sides
    V = fel.V
    msh = fel.msh
    vol = fel.vol
    
    r = SpatialCoordinate(msh)[0]
    #N = 4.0*pi*float(assemble(rhs*r*r*dx(msh)))
    #print "CHECK N",N

    rhs.vector()[:] = 4.0*pi*(rhs.vector())
    
    # Create Dirichlet boundary condition
    dbc = fel.radial_outer_boundary_factory()         
    if boundaryexpr == None:
        boundaryexpr = Constant(0.0)
    bc0 = DirichletBC(V, boundaryexpr, dbc)
    
    # Collect boundary conditions
    bcs = [bc0]

    # Define variational problem
    u = TrialFunction(V)
    v = TestFunction(V)
    #r = V.cell().x[0]
    a = u.dx(0)*v.dx(0)*dx - (2/r)*u.dx(0)*v*dx
    #a = u.dx(0)*v.dx(0)*r*r*dx - 2*r*u.dx(0)*v*dx
    #a = 4.0*pi*u.dx(0)*v.dx(0)*r*r*dx 
    #L = 4.0*pi*rhs*r*r*v*dx 
    L = rhs*v*dx 

    A,b = assemble_system(a, L, bcs)
    if Z != 0:
        print ("APPLYING DELTA",Z)
        # TODO: Should we have delta^3? How do we do that?...
        delta = PointSource(V, Point(fel.rs[0]),4.0*pi*Z)
        delta.apply(b)
    
    print ("= Solving radial poisson equation")
    solve(A, sol.vector(), b)
    print ("= Finished")
    #plot(sol,interactive=True)
     
    #drawgrid = numpy.r_[0:10:0.1]
    #drawgrid = numpy.logspace(5,6,100)
    #drawgrid -= drawgrid[0]
    #gui.plot_radial(drawgrid,rhs)
    #gui.pause()
    #oneoverr = fel.scalar_field_from_expr("Z/x[0]",Z=Z)
    #gui.plot_radial(drawgrid,[sol,oneoverr])
    #gui.plot_radial(drawgrid,sol)
    #gui.pause()
    #exit(0)


    

def solve_radial_helmholtz(fel, sol, rhs, k2, Z=0,boundaryexpr=None):

    V = fel.V
    
    rhs.vector()[:] = -4.0*pi*(rhs.vector())

    if boundaryexpr == None:
        boundaryexpr = Constant(0.0)
    if fel.pbc == [True, True, True]:   
        bcs = []
    else:        
        dbc = fel.dirichlet_boundary_3d_factory()         
        bc0 = DirichletBC(V, boundaryexpr, dbc)
        bcs = [bc0]
    
    # Define variational problem
    u = TrialFunction(V)
    v = TestFunction(V)
    r = V.cell().x[0]
    a = -(inner(nabla_grad(u), nabla_grad(v)) + (1/r)*grad(u)*v*dx - k2 * inner(u, v)) * dx
    L = rhs*v*dx

    A,b = assemble_system(a, L, bcs)
    
    if Z != 0:
        delta = PointSource(V, Point(0),4.0*pi*Z)
        delta.apply(b)
    
    print ("= Solving helmholtz equation")
    solve(A, sol.vector(), b)
    print ("= Finished")
    
    #offset = float(assemble(sol*dx(mesh)))/vol
    #sol.vector()[:] = sol.vector() - offset


def solve_radial_poisson_minus_helmholtz(fel, sol, rhs, k2, deltacoords = [],deltastrengths = [], tmp=None):

    if tmp == None:
        tmp = fel.new_scalar_field()
    
    solve_radial_helmholtz(fel, tmp, rhs, k2, deltacoords,deltastrengths)
    solve_radial_poisson(fel, sol, rhs, deltacoords, deltastrengths)
    sol.vector()[:] = sol.vector() - tmp.vector()
