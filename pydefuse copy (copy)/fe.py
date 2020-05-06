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
from chronicle import chronicle 
import numpy
import gui
import mesher

class FeLattice(object):
    
    def __init__(self,unitcell, pbc, params):        
        self.unitcell = unitcell
        self.pbc = pbc
        self.unitcellinv = numpy.linalg.inv(unitcell)
        self.vol = abs(numpy.dot(numpy.cross(self.unitcell[0],self.unitcell[1]),self.unitcell[2]))
        self.msh = None
        self.V = None
        self.params = params

        set_log_level(WARNING)

        # Setup fenics/dolfin solver parameters
        parameters['allow_extrapolation'] = True
        #parameters["form_compiler"]["log_level"]    = INFO
        parameters["form_compiler"]["cpp_optimize"] = True
        parameters["form_compiler"]["cpp_optimize_flags"] = "-O3"
        parameters["form_compiler"]["optimize"]     = True
        parameters["linear_algebra_backend"] = "PETSc"
        if self.params.threadpar != False:
            parameters["num_threads"] = self.params.threadpar

    def cartesian_to_reduced(self,coords):
        return numpy.dot(coords,self.unitcellinv)
    
    def reduced_to_cartesian(self,coords):
        return numpy.dot(coords,self.unitcell)

    def create_mesh(self,cartcoords,deltastrengths):
        if self.params.mesh_type == 'structured_rectangle':
            divs = self.params.mesh_rectangle_divisions
            self.msh = BoxMesh(Point(0.,0.,0.),Point(1.,1.,1.),divs,divs,divs)
            self.msh.coordinates()[:] = self.reduced_to_cartesian(self.msh.coordinates())

        elif self.params.mesh_type == 'unstructured_rectangle':
            import mshr
            divs = self.params.mesh_rectangle_divisions
            uc = self.unitcell
            p1 = Point(0.0,0.0,0.0)
            p2 = Point(uc[0,0],uc[0,1],uc[0,2])
            p3 = Point(uc[1,0],uc[1,1],uc[1,2])
            p4 = p2+p3
            p5 = Point(uc[2,0],uc[2,1],uc[2,2])
            p6 = p2+p5
            p7 = p3+p5
            p8 = p2+p3+p5
            tetras = [None]*6    
            tetras[0] = [p1, p2, p3, p6]
            tetras[1] = [p2, p3, p4, p6]
            tetras[2] = [p3, p4, p6, p8]
            tetras[3] = [p3, p6, p7, p8]
            tetras[4] = [p3, p5, p6, p7]
            tetras[5] = [p1, p3, p5, p6]
            domain = None
            for tetra in tetras:
                subd = mshr.Tetrahedron(*tetra)
                if domain == None:
                    domain = subd 
                else:
                    domain += subd
            mesh = mshr.generate_mesh(domain, divs, "cgal")
            tree = mesh.bounding_box_tree()            
            for coord in cartcoords:
                p = Point(*coord)    
                (cell_id, dist) = tree.compute_closest_entity(p)
                #print [v.point()[0] for v in vertices(Cell(mesh, cell_id))]
                point_id = min((p.distance(v.point()), v) for v in vertices(Cell(mesh, cell_id)))[1].index()
                mesh.coordinates()[point_id] = coord
            self.msh = mesh
        elif self.params.mesh_type == 'jigzaw':
            uc_divs = self.params.mesh_rectangle_divisions
            sphere_refine = self.params.mesh_mt_refine
            sphere_rad_divs = self.params.mesh_mt_radial_divisions
            self.msh = mesher.jigzaw_icosahedron_mesher(self.unitcell, cartcoords, [0.9]*len(cartcoords), uc_divs, sphere_refine, sphere_rad_divs)
            #self.msh = mesher.jigzaw_icosahedron_mesher(self.unitcell, cartcoords, [0.5]*len(cartcoords), uc_divs, sphere_refine, sphere_rad_divs)
        else:
            raise Exception("mesh_type not understood.")
            

        potential_offset = sum(deltastrengths)/self.vol

        if self.params.prerefine:
            chronicle.start("Prerefine mesh")
            while True:
                chronicle.start("Solve external potential")
                if self.pbc != [False, False, False]:
                    self.V = FunctionSpace(self.msh, "CG", self.params.element_order, constrained_domain=self.periodic_boundary_3d_factory())
                else:
                    self.V = FunctionSpace(self.msh, "CG", self.params.element_order)
    
                rhs = Function(self.V)
                vext = Function(self.V)
                rhs.vector()[:] = 0.0
                vext.vector()[:] = 0.0    
                solve_poisson(self, vext,rhs, cartcoords, deltastrengths)
    
                vext.vector()[:] = vext.vector() - potential_offset
                if self.params.gui:
                    gui.plot1d(self.unitcell,vext,"vext")
                #pylab.waitforbuttonpress()
                chronicle.stop()
            
                chronicle.start("Mark cells")
                cell_markers = CellFunction("bool", self.msh)
                cell_markers.set_all(False)
                refinecount = 0
                minvol = None
                for cell in cells(self.msh):
                    p = cell.midpoint()
                    if (abs(vext(p))**self.params.prerefinepower)*cell.volume() > self.params.prerefine:
                        refinecount+=1
                        cell_markers[cell] = True
                    if minvol == None or minvol > cell.volume():
                        minvol = cell.volume()
                chronicle.stop()
                
                if refinecount > 0:
                    chronicle.start("Do actual mesh refinement")
                    self.msh = refine(self.msh, cell_markers)
                    chronicle.stop()
                else:
                    break
            chronicle.stop()
            vext.vector()[:] = vext.vector() + potential_offset
            self.vext = vext
        else:
            if self.pbc != [False, False, False]:
                self.V = FunctionSpace(self.msh, "CG", self.params.element_order, constrained_domain=self.periodic_boundary_3d_factory())
            else:
                self.V = FunctionSpace(self.msh, "CG", self.params.element_order)

        self.W = MixedFunctionSpace([self.V,self.V])
        #plot(self.msh)
        #interactive()
            
    def refine_mesh(self, field, field_grad, field_lapl, eps):
        # A square "blub" of volume L*L*L with laplacian at center = a
        # is described by:
        #   -(8/3*a/L^4)*(x-L/2)(x+L/2)(y-L/2)(y+L/2)(z-L/2)(z+L/2)
        # which integrates to
        #   a*(4/99)*L^5/99 = a*4/99*(L^3)^(5/3) = a*(4/99)*V^(5/3)
        # We want the integration of this 'error' over all of space
        # to contribute less than epsilon. Hence, our error estimate is:
        #   a*(4/99)*V^(5/3)*Vtot/V = a*(4/99)*V^(2/3)*Vtot < eps
        
        chronicle.start("Detecting grid cells to refine")
        cell_markers = CellFunction("bool", self.msh)
        cell_markers.set_all(False)
        count = 0
        
        #errest = lapln*(4.0/99.0)*0.02**(2.0/3.0)*vol
        #errest = project(errest,V)
        #plot1d(errest,"Error estimate")
    
        for cell in cells(self.msh):
            p = cell.midpoint()
            #if (p.x()-1.0)**2+(p.y()-2.5)**2+(p.z()-2.5)**2 < 0.25:
            #    print "P",p.x(),p.y(),p.z(),abs(lapln(p)*(4.0/99.0)*cell.volume()**(2.0/3.0)*vol)
            lval = min(abs(field_lapl(p)),10)
            if (lval*(4.0/99.0)*cell.volume()**(2.0/3.0)*self.vol) > eps:
                cell_markers[cell] = True
                count+=1   

        chronicle.stop()

        chronicle.start("Refining grid")
        self.msh = refine(self.msh, cell_markers)
        chronicle.stop()
        self.V = FunctionSpace(self.msh, "CG", self.params.element_order, constrained_domain=self.periodic_boundary_3d_factory())
        self.W = MixedFunctionSpace([self.V,self.V])
        # Clear affected quantities
        self.vext = None

    def refine_mesh_integratedvalue(self, field, eps):
        # Refine cells whose present value is > eps
        
        chronicle.start("Detecting grid cells to refine")
        cell_markers = CellFunction("bool", self.msh)
        cell_markers.set_all(False)
        count = 0
            
        for cell in cells(self.msh):
            p = cell.midpoint()
            if abs(field(p)*cell.volume()) > eps:
                cell_markers[cell] = True
                count+=1   

        chronicle.stop()

        chronicle.start("Refining grid")
        self.msh = refine(self.msh, cell_markers)
        chronicle.stop()
        self.V = FunctionSpace(self.msh, "CG", self.params.element_order, constrained_domain=self.periodic_boundary_3d_factory())
        self.W = MixedFunctionSpace([self.V,self.V])
        # Clear affected quantities
        self.vext = None        
        
    def transfer_scalar_field(self, field):
        return interpolate(field,self.V)

    def transfer_mixed_scalar_field(self, field):
        return interpolate(field,self.W)

    def new_scalar_field(self):
        return Function(self.V)

    def new_mixed_scalar_field(self):
        return Function(self.W)

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
        VV = VectorFunctionSpace(self.fe.msh, 'CG', self.params.element_order)
        return Function(VV)

    def integrate_scalar_field(self,field):
        return float(assemble((field)*dx(self.msh)))

    def grad_lapl(self,field):
        #VV = VectorFunctionSpace(self.msh, 'CG', 1) 
        grad_field2 = Function(self.V)
        lapl_field = Function(self.V)

        # Create Dirichlet boundary condition
        dbc = self.dirichlet_boundary_3d_factory()         
        boundaryexpr = Constant(0.0)
        if self.pbc == [True, True, True]:   
            bc0 = DirichletBC(self.V, boundaryexpr, dbc, method="pointwise")
        else:
            bc0 = DirichletBC(self.V, boundaryexpr, dbc)
        
        # Collect boundary conditions
        bcs = [bc0]
                
        u = TrialFunction(self.V)
        v = TestFunction(self.V)        
        a = u*v*dx
        L = dot(grad(field),grad(field))*v*dx
        solve(a == L, grad_field2,bcs)

        u = TrialFunction(self.V)
        v = TestFunction(self.V)        
        a = u*v*dx
        L = -dot(grad(field),grad(v))*dx
        solve(a == L, lapl_field,bcs)

        return grad_field2, lapl_field

    def grad_lapl_alt(self,field):
            grad_field2 = project(dot(grad(field),grad(field)),self.V)
            lapl_field = project(div(grad(field)),self.V)            
            return grad_field2, lapl_field

    def vh_to_dens(self,vh,dens,N):
        bcs = []
        
        r = SpatialCoordinate(self.msh)[0]
        u = TrialFunction(self.V)
        v = TestFunction(self.V)        
        a = u*v*dx
        L = (1.0/(4.0*pi))*inner(grad(vh),grad(v))*dx
        solve(a == L, dens,bcs)
        dens.vector()[:] = N/self.vol + dens.vector()[:]
        
    def periodic_boundary_3d_factory(self):
    
        lattice = self
        if self.pbc == [True, True, True]:
    
            class PeriodicBoundary3D(SubDomain):
                
                def inside(self, x, on_boundary):
                    xr = lattice.cartesian_to_reduced(x)
                    return bool(
                        (
                            (near(x[0], 0) or near(x[1], 0) or near(x[2], 0)) and
                            (not ( (near(xr[0], 1.)) or (near(xr[1], 1.)) or (near(xr[2], 1.)) ))
                        ) and on_boundary)
            
                def map(self, x, y):
                    xr = lattice.cartesian_to_reduced(x)
                    y[0] = x[0]
                    y[1] = x[1]
                    y[2] = x[2]
                    if near(xr[0], 1.0):
                        y[0] -= lattice.unitcell[0][0]
                        y[1] -= lattice.unitcell[0][1]
                        y[2] -= lattice.unitcell[0][2]
                    if near(xr[1], 1.0):
                        y[0] -= lattice.unitcell[1][0]
                        y[1] -= lattice.unitcell[1][1]
                        y[2] -= lattice.unitcell[1][2]
                    if near(xr[2], 1.0):
                        y[0] -= lattice.unitcell[2][0]
                        y[1] -= lattice.unitcell[2][1]
                        y[2] -= lattice.unitcell[2][2]
            return PeriodicBoundary3D()    

        else:
            pbc = self.pbc

            class SemiPeriodicBoundary3D(SubDomain):
                
                def inside(self, x, on_boundary):
                    xr = lattice.cartesian_to_reduced(x)
                    return bool(
                        (
                            ((not pbc[0] or near(x[0], 0)) or (not pbc[1] or near(x[1], 0)) or (not pbc[2] or near(x[2], 0))) and
                            (not ( (pbc[0] and near(xr[0], 1.)) or (pbc[1] and near(xr[1], 1.)) or (pbc[2] and near(xr[2], 1.)) ))
                        ) and on_boundary)
            
                def map(self, x, y):
                    xr = lattice.cartesian_to_reduced(x)
                    y[0] = x[0]
                    y[1] = x[1]
                    y[2] = x[2]
                    if pbc[0] and near(xr[0], 1.0):
                        y[0] -= lattice.unitcell[0][0]
                        y[1] -= lattice.unitcell[0][1]
                        y[2] -= lattice.unitcell[0][2]
                    if pbc[1] and near(xr[1], 1.0):
                        y[0] -= lattice.unitcell[1][0]
                        y[1] -= lattice.unitcell[1][1]
                        y[2] -= lattice.unitcell[1][2]
                    if pbc[2] and near(xr[2], 1.0):
                        y[0] -= lattice.unitcell[2][0]
                        y[1] -= lattice.unitcell[2][1]
                        y[2] -= lattice.unitcell[2][2]
            
            return SemiPeriodicBoundary3D()    

    def dirichlet_boundary_3d_factory(self):    
        if self.pbc == [True, True, True]:
            class DirichletBoundary3D(SubDomain):        
                def inside(self, x, on_boundary):
                    return bool(((near(x[0], 0) and near(x[1], 0) and near(x[2], 0))))       
            return DirichletBoundary3D()
        else:
            pbc = self.pbc
            lattice = self
            class SemiDirichletBoundary3D(SubDomain):        
                def inside(self, x, on_boundary):
                    xr = lattice.cartesian_to_reduced(x)
                    return bool(   (not pbc[0] and (near(xr[0], 0) or near(xr[0], 1.0)) )
                                or (not pbc[1] and (near(xr[1], 0) or near(xr[1], 1.0)) )
                                or (not pbc[2] and (near(xr[2], 0) or near(xr[2], 1.0)) )  )
            return SemiDirichletBoundary3D()
        
def solve_poisson(fel, sol, rhs, deltacoords = [], deltastrengths = [], boundaryexpr=None, preserve_rhs=True):
    # Note, this implementation have the sign changed both on left and right hand sides
    # Note, IMPORTANT, rhs is CHANGED 
    chronicle.start("Solve Poisson's equation")

    V = fel.V
    msh = fel.msh
    vol = fel.vol

    if fel.pbc == [True, True, True]:
        chronicle.start("Adjust rhs offset")
        offset = float(assemble(rhs*dx(msh))) + sum(deltastrengths)
        rhs.vector()[:] = 4.0*pi*(rhs.vector() - offset/vol)
        chronicle.stop()
    
    # Create Dirichlet boundary condition
    dbc = fel.dirichlet_boundary_3d_factory()         
    if boundaryexpr == None:
        boundaryexpr = Constant(0.0)
    if fel.pbc == [True, True, True]:   
        bc0 = DirichletBC(V, boundaryexpr, dbc, method="pointwise")
    else:
        bc0 = DirichletBC(V, boundaryexpr, dbc)
    
    # Collect boundary conditions
    bcs = [bc0]
    
    # Define variational problem
    u = TrialFunction(V)
    v = TestFunction(V)
    a = dot(grad(u), grad(v))*dx
    L = rhs*v*dx

    chronicle.start("Assembling system")
    A,b = assemble_system(a, L, bcs)
    
    for i in range(len(deltastrengths)):
        delta = PointSource(V, Point(deltacoords[i]),4.0*pi*deltastrengths[i])
        delta.apply(b)
    chronicle.stop()
    
    chronicle.start("Run solver")
    solve(A, sol.vector(), b)
    chronicle.stop()
    
    if fel.pbc == [True, True, True]:
        chronicle.start("Adjusting solution offset")
        offset = float(assemble(sol*dx(msh)))/vol
        sol.vector()[:] = sol.vector() - offset
        chronicle.stop()

    if preserve_rhs:
        chronicle.start("Transform rhs back to preserve it")
        rhs.vector()[:] = rhs.vector()/4.0*pi + offset/vol
        chronicle.stop()

    chronicle.stop()

def solve_helmholtz(fel, sol, rhs, k2, deltacoords = [], deltastrengths = [], boundaryexpr=None):
    chronicle.start("Solve helmholtz' equation")

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
    
    chronicle.start("Assembling system")
    # Define variational problem
    u = TrialFunction(V)
    v = TestFunction(V)
    a = -(inner(nabla_grad(u), nabla_grad(v)) - k2 * inner(u, v)) * dx
    L = rhs*v*dx

    A,b = assemble_system(a, L, bcs)
    chronicle.stop()
    
    for i in range(len(deltastrengths)):
        delta = PointSource(V, Point(deltacoords[i]),-4.0*pi*deltastrengths[i])
        delta.apply(b)
    
    chronicle.start("Run solver")
    solve(A, sol.vector(), b)
    chronicle.stop()

    chronicle.stop()

def solve_poisson_minus_helmholtz(fel, sol, rhs, k2, deltacoords = [],deltastrengths = [], tmp=None):
    # TODO: Rewrite as a single differential equation solution
    chronicle.start("Solve Poisson's equation minus Helmholtz'")

    if tmp == None:
        tmp = fel.new_scalar_field()
    
    solve_helmholtz(fel, tmp, rhs, k2, deltacoords,deltastrengths)
    solve_poisson(fel, sol, rhs, deltacoords, deltastrengths)
    sol.vector()[:] = sol.vector() - tmp.vector()
    
    chronicle.stop()