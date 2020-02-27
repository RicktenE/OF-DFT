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

import math
import numpy, pylab
from pydefuse.chronicle import chronicle
from dolfin import *

from pydefuse import feradial, func_tf
from densityobj import DensityFields, DensityRadialWeakForm
import gui
from pydefuse.feradial import solve_radial_poisson

class OfdftRadial(object):

    CF=3.0/10.0*(3.0*math.pi**2)**(2.0/3.0)
    CX=3.0/4.0*(3.0/math.pi)**(1.0/3.0)
    hartree2ev=27.21138386

    def __init__(self, rs, Z, N, functionals, **params):  
        self.rs = rs
        self.Z = Z
        self.functionals = functionals
        if N == None:
            self.N = Z
        else:
            self.N = N         
    
        # Defaults
        self.params = { 
          #basestep = 0.1
          'basestep':0.05,
          #'prerefine':0.01,
          #'prerefinepower':2.0,
          #prerefine = 0.005
          #prerefinepower = 1.0
          'prerefine': 0.1,
          'prerefinepower': 1.0,
          'dynrefine':False,
          'threadpar':False,
          'cuspcond':True,
          'eps':1e-4                       
        }
        self.params.update(params)

    def functional_energy_density_expr(self):
        #tfenergy = -(3.0/4.0)*(3.0/pi)**(1.0/3.0)*abs(n)**(5.0/3.0)
        #tfpot = (5.0/3.0)*CF*abs(n)**(2.0/3.0)
        #functional.vector()[:] = (5.0/3.0)*CF*sqrt(n.vector().inner(n.vector()))**(2.0/3.0)
    
        #field.vector()[:] = (3.0/4.0)*(3.0/pi)**(1.0/3.0)*self.CF*numpy.power(self.n.vector()[:],(5.0/3.0))
        return self.CF*pow(self.n,(5.0/3.0))
        
        ##diracenergy=CX*abs(n)**(4.0/3.0)
        #diracpot = -(4.0)/(3.0)*CX*abs(n)**(1.0/3.0)
    
        #lmbd = 1.0/9.0
        ##weisackerenergy = lamba*(1.0/8.0*gradn2/n)
        #weisackerpot = lmbd*(1.0/8.0*gradn2/(n**2+1.0) - 1.0/4.0*lapln/(abs(n)+1.0))

    
    def functional_energy_density(self,field):
        #tfenergy = -(3.0/4.0)*(3.0/pi)**(1.0/3.0)*abs(n)**(5.0/3.0)
        #tfpot = (5.0/3.0)*CF*abs(n)**(2.0/3.0)
        #functional.vector()[:] = (5.0/3.0)*CF*sqrt(n.vector().inner(n.vector()))**(2.0/3.0)
    
        #field.vector()[:] = (3.0/4.0)*(3.0/pi)**(1.0/3.0)*self.CF*numpy.power(self.n.vector()[:],(5.0/3.0))
        field.vector()[:] = self.CF*numpy.power(self.n.vector()[:],(5.0/3.0))
        
        ##diracenergy=CX*abs(n)**(4.0/3.0)
        #diracpot = -(4.0)/(3.0)*CX*abs(n)**(1.0/3.0)
    
        #lmbd = 1.0/9.0
        ##weisackerenergy = lamba*(1.0/8.0*gradn2/n)
        #weisackerpot = lmbd*(1.0/8.0*gradn2/(n**2+1.0) - 1.0/4.0*lapln/(abs(n)+1.0))
    def functional_potential(self,field):
        #tfenergy = -(3.0/4.0)*(3.0/pi)**(1.0/3.0)*abs(n)**(5.0/3.0)
        #tfpot = (5.0/3.0)*CF*abs(n)**(2.0/3.0)
        #functional.vector()[:] = (5.0/3.0)*CF*sqrt(n.vector().inner(n.vector()))**(2.0/3.0)
    
        field.vector()[:] = (5.0/3.0)*self.CF*numpy.power(self.n.vector()[:],(2.0/3.0))
        
        ##diracenergy=CX*abs(n)**(4.0/3.0)
        #diracpot = -(4.0)/(3.0)*CX*abs(n)**(1.0/3.0)
    
        #lmbd = 1.0/9.0
        ##weisackerenergy = lamba*(1.0/8.0*gradn2/n)
        #weisackerpot = lmbd*(1.0/8.0*gradn2/(n**2+1.0) - 1.0/4.0*lapln/(abs(n)+1.0))
     
    def initialize(self):
        timer.start("Initialize system")
        self.felr = feradial.FeRadialLattice(self.rs, self.params)

        print "Gridpoints:",self.felr.num_vertices,"range:",self.felr.grid_start,"-",self.felr.grid_end
        print "Atom:", self.Z
        print "Volume:",self.felr.vol
        print "Number of electrons in system:",self.N
                
        timer.stop()

    def prepare_starting_mesh(self):
        timer.start("Prepare mesh")
        self.felr.create_mesh(self.Z)
        timer.stop()
        #plot(self.felr.msh,interactive = True)        
            
    def prepare_starting_density(self):
        timer.start("Prepare starting density")
        n = self.felr.scalar_field_from_expr('rho0*exp(-fac*Z*x[0])',Z=self.Z,fac=1.8,rho0=0.5*self.Z**3)
        intn = self.felr.integrate_scalar_field(n)
        print "Density integrated before adjustment:",intn
        #n.vector()[:] = n.vector() - intn/vol + N/vol
        n.vector()[:] = n.vector()*self.N/intn  
        intn = self.felr.integrate_scalar_field(n)
        print "Density integrated after first adjustment:",intn
        self.n = n
        timer.stop()

    def solve_tf(self):
        fel = self.felr
        V = fel.V
        msh = fel.msh
        vol = fel.vol

        self.n = self.felr.new_scalar_field()
        sf = self.felr.new_scalar_field()
        self.tfvh = self.felr.new_scalar_field()
        
        timer.start("Solve unmodified TF equation")
        feradial.solve_radial_tf(self.felr, self.n, self.Z, screening_function=sf)
        timer.stop()

        timer.start("Calculations etc.")
        N = self.felr.integrate_scalar_field(self.n)
        print "CHECK N = int n_tf(r)",N
        self.iters = 0
        #r = SpatialCoordinate(msh)[0]
        #self.tfvh2 = project(-self.Z/r*sf + self.Z/r,V)
        #feradial.solve_radial_poisson(fel, self.tfvh, self.tfn)
        timer.stop()        

        nvec = self.n.vector()
        negs =(nvec<0.0).sum()
        minval = nvec.min()
        print "Number of negative density elements:",negs,"min value",minval
        #if minval < 0.0:
        #    nvec[:] = nvec - minval
        if negs > 0:
            nvec[nvec<0] = 0.0


    def solve_newton_vh(self):

        fel = self.felr
        V = fel.V
        msh = fel.msh
        vol = fel.vol
        Z = Constant(self.Z)

        alpha = 4.0/1.0*(2.0*self.Z/(9.0*pi**2))**(1.0/3.0)
        mu = 1.0
        
        def inner_boundary(x, on_boundary):
            return on_boundary and near(x[0],self.rs[0])
        
        def outer_boundary(x, on_boundary):
            return on_boundary and near(x[0],self.rs[-1])

        u_k = interpolate(Constant(-1.0), V)  # previous (known) u
        n = Function(V)
        gr2 = Function(V)
        la = Function(V)
        
        u = TrialFunction(V)
        v = TestFunction(V)

        bcs_du = []

        du_ = TrialFunction(V)

        du = Function(V)
        u_ = Function(V)  # u = u_k + omega*du
        omega = self.params['basestep']       # relaxation parameter
        eps = 1.0
        self.iters = 0
        maxiter = 5000
        r = SpatialCoordinate(msh)[0]
        while eps > self.params['eps'] and self.iters < maxiter:
            self.iters += 1

            if self.params['cuspcond']:

                timer.start("Preparing cusp correction")

                #timer.start("Cusp correction, fixing negative density and correct for lost/added electrons")

                #r = SpatialCoordinate(self.felr.msh)[0]
                #gr = project(u_k.dx(0),V)
                #n = project(-1.0/(4.0*pi)*(gr.dx(0)+2/r*gr),V)
                #n0 = n(self.rs[1])
                n0 = n(0)
                                
                #print "************ n(0)",n0/self.Z**3,":",n0,n(0),n2(0),":",n(self.rs[1]),n2(self.rs[1])
                if n0 <= 0.0:
                    #potexpr = -self.Z/r
                    k = Constant(sqrt(5.0/9.0*self.CF)*(0.5*self.Z**3)**(1.0/3.0))
                    potexpr = (-self.Z/r * (1 - exp(-2*k*r)) - k*self.Z*exp(-2*k*r))
                    print "== k (estimated):",sqrt(5.0/9.0*self.CF)*(0.5*self.Z**3)**(1.0/3.0)
                else:    
                    k = Constant(sqrt(5.0/9.0*self.CF)*n0**(1.0/3.0))
                    print "== k value used:",sqrt(5.0/9.0*self.CF)*n0**(1.0/3.0)
                    potexpr = self.felr.scalar_field_from_expr('(x[0] != 0.0)?(-Z/x[0] * (1.0 - exp(-2.0*k*x[0])) - k*Z*exp(-2.0*k*x[0])):(-3*k*Z)',k=k,Z=self.Z)
                    #potexpr = (-Z/r * (1 - exp(-2*k*r)) - k*Z*exp(-2*k*r))

                    #potexpr = project(Expression('(x[0] != 0.0)?(-Z/x[0] * (1.0 - exp(-2.0*k*x[0])) - k*Z*exp(-2.0*k*x[0])):(-k*Z-2.0*k*Z)',k=k,Z=self.Z),V)

#                     pylab.clf()
#                     drawgrid = numpy.logspace(-7,0,100)
#                     #drawgrid -= drawgrid[0]
#                     drawgrid -= drawgrid[0] - self.rs[1]
#                     yvals = [n(x) for x in drawgrid]
#                     yvals2 = [n2(x) for x in drawgrid]
#                     yvals3 = [n3(x) for x in drawgrid]
#                     yvals4 = [u_k(x) for x in drawgrid]
#                     pylab.semilogx(drawgrid,yvals,'bx-')
#                     pylab.semilogx(drawgrid,yvals2,'r.--')
#                     pylab.semilogx(drawgrid,yvals3,'g.--')
#                     pylab.semilogx(drawgrid,yvals4,'y.--')
#                     pylab.title("n")
#                     print "WHAT",n3(0.0),self.rs[1],u_k(0.0)
#                     gui.pause()
                    #pylab.pause(0.00001)
                    
                timer.stop()                                
            else:   
                potexpr = -self.Z/r

            timer.start("Calculate functionals")
            funcpots = 0
            #densobj = DensityFields(n,gr2,la)
            ## TODO, if tf already in potentials, skip that term instead
            #for f in self.functionals:
            #    funcpots += f.potential_enhancement_expr(densobj)
            #timer.stop()

            # The adjustment of mu isn't very fundamental, there will be a component of mu hidden as an offset in v_h,
            # but this doesn't matter, unless one want an actual value for mu
            timer.start("Handle chemical potential")
            u_i = project(Constant(mu) -potexpr - u_k,V)
            minval = u_i.vector().min()
            if minval < 0.0:
                #u_k.vector()[:]=-minval
                mu-=minval-1e-14
            timer.stop()    

            u_i = project(Constant(mu) -potexpr - u_k,V)
            print "MINIMUM VALUE",u_i.vector().min()

            timer.start("Solve for hartree potential")
            r = SpatialCoordinate(self.felr.msh)[0]
            #F = -u_k.dx(0)*v.dx(0)*dx + (2/r)*u_k.dx(0)*v*dx + 8.0*sqrt(2.0)/(3.0*pi)*(sqrt(Constant(mu) - potexpr - u_k))**3*v*dx
            #print "::::::::::",funcpots
            #funcpots = (1+(-4.0/5.0)*(self.CX/self.CF)*1.0/pow(u_k,(1.0/3.0)))
            #funcpots = (-4.0/3.0)*self.CX*pow(n,1.0/3.0)
            F = -u_k.dx(0)*v.dx(0)*dx + (2/r)*u_k.dx(0)*v*dx + 8.0*sqrt(2.0)/(3.0*pi)*(sqrt(Constant(mu) - potexpr - u_k))**3*v*dx

            J = derivative(F, u_k, du_)
            A, b = assemble_system(J, -F, bcs_du)
            solve(A, du.vector(), b)
            timer.stop()

            #pylab.clf()
            #drawgrid = numpy.logspace(-5,2,100)
            #vals = [du(r) for r in drawgrid]
            #pylab.plot(drawgrid,yvals,'bx-')
            #pylab.title("dn")
            #pylab.waitforbuttonpress()
            
            timer.start("Calculating error")
            avg = sum(du.vector().array())/len(du.vector().array())
            eps = numpy.linalg.norm(du.vector().array()-avg, ord=numpy.Inf)
            print 'Iter',self.iters,'norm:', eps
            timer.stop()

            timer.start("calculate electron density (step1)")
            n1 = project(8.0/(4.0*pi)*sqrt(2.0)/(3.0*pi)*(sqrt(Constant(mu) - potexpr - u_k))**3,V)
            timer.stop()    
            
            timer.start("Taking hartree potential step")
            u_.vector()[:] = u_k.vector() + omega*du.vector()
            u_k.assign(u_)
            timer.stop()    

            timer.start("calculate electron density (step2)")
            r = SpatialCoordinate(self.felr.msh)[0]
            #gr = project(u_k.dx(0),V)
            #n2 = project(-1.0/(4.0*pi)*(gr.dx(0)+2/r*gr),V)
            n2 = Function(V)
            self.felr.vh_to_dens(u_k,n2)
            timer.stop()    

            timer.start("interpolate between step1 and step2 densities")
            n = project((1.0-exp(-100.0*n1**2))*n1 + exp(-100.0*n1**2)*n2,V)
            #n = n1
            nvec = n.vector()
            nvec[nvec<0.0]=0.0
            timer.stop()    

            timer.start("calculate electron density gradient and laplacian")
            gr2, la = self.felr.grad_lapl(n)
            #r = SpatialCoordinate(self.felr.msh)[0]
            #gr = project(u_k.dx(0),V)
            #n2 = project(-1.0/(4.0*pi)*(gr.dx(0)+2/r*gr),V)
            timer.stop()    

#             pylab.clf()
#             drawgrid = numpy.logspace(-5,2,100)
#             #drawgrid -= drawgrid[0]
#             yvals = [gr2(r) for r in drawgrid]
#             yvals2 = [la(r) for r in drawgrid]
#             #yvals2 = [self.n(r) for r in drawgrid]
#             #pylab.semilogx(drawgrid,yvals,'bx-')
#             pylab.semilogx(drawgrid,yvals2,'ro-')
#             #pylab.loglog(drawgrid,yvals2,'k.-')
#             pylab.title("grad lapl")
#             pylab.pause(0.0001)
 
            pylab.clf()
            drawgrid = numpy.logspace(-5,2,100)
            yvals = [n(rr) for rr in drawgrid]
            pylab.loglog(drawgrid,yvals,'bx-')
            pylab.title("density")
            pylab.pause(0.0001)
 
#             pylab.clf()
#             drawgrid = numpy.logspace(-5,2,100)
#             yvals = [u_k(r) for r in drawgrid]
#             pylab.semilogx(drawgrid,yvals,'bx-')
#             pylab.title("density")
#             pylab.pause(0.0001)
           
        self.n = n








    def solve_newton_vh_2(self):

        fel = self.felr
        V = fel.V
        msh = fel.msh
        vol = fel.vol
        Z = Constant(self.Z)

        alpha = 4.0/1.0*(2.0*self.Z/(9.0*pi**2))**(1.0/3.0)
        mu = 1.0
        
        def inner_boundary(x, on_boundary):
            return on_boundary and near(x[0],self.rs[0])
        
        def outer_boundary(x, on_boundary):
            return on_boundary and near(x[0],self.rs[-1])

        u_k = interpolate(Constant(-1.0), V)  # previous (known) u
        n = Function(V)
        n1 = Function(V)
        gr2 = Function(V)
        la = Function(V)
        
        u = TrialFunction(V)
        v = TestFunction(V)

        bcs_du = []

        du_ = TrialFunction(V)

        du = Function(V)
        u_ = Function(V)  # u = u_k + omega*du
        omega = self.params['basestep']       # relaxation parameter
        eps = 1.0
        self.iters = 0
        maxiter = 5000
        while eps > self.params['eps'] and self.iters < maxiter:
            self.iters += 1
            r = SpatialCoordinate(msh)[0]
            if self.params['cuspcond']:
                timer.start("Preparing cusp correction")
                n0 = n(0)
                if n0 <= 0.0:
                    k = Constant(sqrt(5.0/9.0*self.CF)*(0.5*self.Z**3)**(1.0/3.0))
                    potexpr = (-self.Z/r * (1 - exp(-2*k*r)) - k*self.Z*exp(-2*k*r))
                    print "== k (estimated):",sqrt(5.0/9.0*self.CF)*(0.5*self.Z**3)**(1.0/3.0)
                else:    
                    k = Constant(sqrt(5.0/9.0*self.CF)*n0**(1.0/3.0))
                    print "== k value used:",sqrt(5.0/9.0*self.CF)*n0**(1.0/3.0)
                    potexpr = self.felr.scalar_field_from_expr('(x[0] != 0.0)?(-Z/x[0] * (1.0 - exp(-2.0*k*x[0])) - k*Z*exp(-2.0*k*x[0])):(-3*k*Z)',k=k,Z=self.Z)
                    
                timer.stop()                                
            else:   
                potexpr = -self.Z/r

            timer.start("Calculate functionals")
            funcpots = 0

            densobj = DensityFields(n1,gr2,la)
            ## TODO, if tf already in potentials, skip that term instead
            funcpots -= func_tf.potential_expr(densobj)
            for f in self.functionals:
                funcpots += f.potential_expr(densobj)
            timer.stop()

            # The adjustment of mu isn't very fundamental, there will be a component of mu hidden as an offset in v_h,
            # but this doesn't matter, unless one want an actual value for mu
            timer.start("Handle chemical potential")
            u_i = project(Constant(mu) - funcpots -potexpr - u_k,V)
            minval = u_i.vector().min()
            print "Minval",minval
            if minval < 0.0:
                #u_k.vector()[:]=-minval
                mu-=minval-1e-10
            timer.stop()    

            u_i = project(Constant(mu) - funcpots -potexpr - u_k,V)
            print "MINIMUM VALUE",u_i.vector().min()

            timer.start("Solve for hartree potential")
            r = SpatialCoordinate(self.felr.msh)[0]
            #F = -u_k.dx(0)*v.dx(0)*dx + (2/r)*u_k.dx(0)*v*dx + 8.0*sqrt(2.0)/(3.0*pi)*(sqrt(Constant(mu) - potexpr - u_k))**3*v*dx
            #print "::::::::::",funcpots
            #funcpots = (1+(-4.0/5.0)*(self.CX/self.CF)*1.0/pow(u_k,(1.0/3.0)))
            #funcpots = (-4.0/3.0)*self.CX*pow(n,1.0/3.0)
            F = -u_k.dx(0)*v.dx(0)*dx + (2/r)*u_k.dx(0)*v*dx + 8.0*sqrt(2.0)/(3.0*pi)*(sqrt(Constant(mu) - funcpots - potexpr - u_k))**3*v*dx

            J = derivative(F, u_k, du_)
            A, b = assemble_system(J, -F, bcs_du)
            solve(A, du.vector(), b)
            timer.stop()

            pylab.clf()
            drawgrid = numpy.logspace(-5,2,100)
            yvals = [du(r) for r in drawgrid]
            pylab.plot(drawgrid,yvals,'bx-')
            pylab.title("dn")
            pylab.waitforbuttonpress()
            
            timer.start("Calculating error")
            avg = sum(du.vector().array())/len(du.vector().array())
            eps = numpy.linalg.norm(du.vector().array()-avg, ord=numpy.Inf)
            print 'Iter',self.iters,'norm:', eps
            timer.stop()

            # The adjustment of mu isn't very fundamental, there will be a component of mu hidden as an offset in v_h,
            # but this doesn't matter, unless one want an actual value for mu
            timer.start("Handle chemical potential")
            u_i = project(Constant(mu) - funcpots -potexpr - u_k,V)
            minval = u_i.vector().min()
            print "HUH",minval
            if minval < 0.0:
                #u_k.vector()[:]=-minval
                mu-=minval-1e-14
            timer.stop()    

            timer.start("calculate electron density (step1)")
            n1 = project(8.0/(4.0*pi)*sqrt(2.0)/(3.0*pi)*(sqrt(Constant(mu) - funcpots - potexpr - u_k))**3,V)
            timer.stop()    
            
            timer.start("Taking hartree potential step")
            u_.vector()[:] = u_k.vector() + omega*du.vector()
            u_k.assign(u_)
            timer.stop()    

            timer.start("calculate electron density (step2)")
            r = SpatialCoordinate(self.felr.msh)[0]
            #gr = project(u_k.dx(0),V)
            #n2 = project(-1.0/(4.0*pi)*(gr.dx(0)+2/r*gr),V)
            n2 = Function(V)
            self.felr.vh_to_dens(u_k,n2)
            timer.stop()    

            timer.start("interpolate between step1 and step2 densities")
            n = project((1.0-exp(-100.0*n1**2))*n1 + exp(-100.0*n1**2)*n2,V)
            #n = n1
            nvec = n.vector()
            nvec[nvec<0.0]=0.0
            timer.stop()    

            timer.start("calculate electron density gradient and laplacian")
            gr2, la = self.felr.grad_lapl(n)
            timer.stop()    

#             pylab.clf()
#             drawgrid = numpy.logspace(-5,2,100)
#             #drawgrid -= drawgrid[0]
#             yvals = [gr2(r) for r in drawgrid]
#             yvals2 = [la(r) for r in drawgrid]
#             #yvals2 = [self.n(r) for r in drawgrid]
#             #pylab.semilogx(drawgrid,yvals,'bx-')
#             pylab.semilogx(drawgrid,yvals2,'ro-')
#             #pylab.loglog(drawgrid,yvals2,'k.-')
#             pylab.title("grad lapl")
#             pylab.pause(0.0001)
 
            pylab.clf()
            drawgrid = numpy.logspace(-5,2,100)
            yvals = [n(r) for r in drawgrid]
            pylab.loglog(drawgrid,yvals,'bx-')
            pylab.title("density")
            pylab.pause(0.0001)
 
#             pylab.clf()
#             drawgrid = numpy.logspace(-5,2,100)
#             yvals = [u_k(r) for r in drawgrid]
#             pylab.semilogx(drawgrid,yvals,'bx-')
#             pylab.title("density")
#             pylab.pause(0.0001)
           
        self.n = n








    def solve_newton_n(self):

        fel = self.felr
        V = fel.V
        msh = fel.msh
        vol = fel.vol

        alpha = 4.0/1.0*(2.0*self.Z/(9.0*pi**2))**(1.0/3.0)
        
        def inner_boundary(x, on_boundary):
            return on_boundary and near(x[0],self.rs[0])
        
        def outer_boundary(x, on_boundary):
            return on_boundary and near(x[0],self.rs[-1])
        
        #u_k = Function(V)
        #self.prepare_starting_density()
        #u_k.assign(self.n)
        #u_k.vector()[:]=u_k.vector() + 0.1
        u_k = self.felr.scalar_field_from_expr('rho0*(end-x[0])',rho0=0.5*self.Z**3,end=self.rs[-1])
        #u_k = self.felr.scalar_field_from_expr('rho0*pow(x[0],(-3.0/2.0))',Z=self.Z,fac=1.8,rho0=0.1*self.Z**3)
        intn = self.felr.integrate_scalar_field(u_k)
        print "Density integrated before adjustment:",intn
        u_k.vector()[:] = u_k.vector()*self.N/intn  
        intn = self.felr.integrate_scalar_field(u_k)
        print "Density integrated after first adjustment:",intn        
        
        #u_k = interpolate(Constant(self.Z/self.felr.vol), V)  # previous (known) u
        u_k_pre = Function(V)
        v_h = Function(V)
        rhs = Function(V)
        tmp = Function(V)
       
        u = TrialFunction(V)
        v = TestFunction(V)

        du_ = TrialFunction(V)

        du = Function(V)
        u_ = Function(V)  # u = u_k + omega*du
        omega = 0.1       # relaxation parameter
        eps = 1.0
        tol = 1.0E-6
        self.iters = 0
        maxiter = 200
        while eps > tol and self.iters < maxiter:
            self.iters += 1

            rhs.assign(u_k)
            feradial.solve_radial_poisson(self.felr,v_h,rhs,Z=0,boundaryexpr=self.felr.expression("Z/x[0]",Z=self.Z))

            #nvec = u_k.vector()
            #negs =(nvec<0.0).sum()
            #minval = nvec.min()
            #print "** Number of negative density elements:",negs,"min value",minval

            potexpr = 0
            timer.start("Calculate functionals")
            densobj = DensityRadialWeakForm(u_k,v)
            for f in self.functionals:
                potexpr += f.potential_weakform(densobj)
            timer.stop()

            # Why is this needed?!
            r = SpatialCoordinate(msh)[0]
            #t = Function(V)
            #t = project(v_h - self.Z/r,V)

            #pylab.clf()
            #drawgrid = numpy.logspace(0.1,1.2,100)
            #drawgrid -= drawgrid[0]
            #yvals = [-t(r) for r in drawgrid]
            #yvals2 = [v_h(r) for r in drawgrid]
            #pylab.loglog(drawgrid,yvals,'bx-')
            #pylab.loglog(drawgrid,yvals2,'ro-')
            #pylab.title("v_H")
            #gui.pause()

            F = potexpr*dx + (-self.Z/r + v_h)*v*dx
            #F = ((5.0/3.0)*self.CF*pow(u_k,(2.0/3.0)) + t)*v*dx

            #bce = Constant(0.0)
            ##bce = 1.0/(self.rs[-1]**6)
            #bc0 = DirichletBC(V, bce, outer_boundary)
            #bcs_du = [bc0]
            bcs_du = []

            J = derivative(F, u_k, du_)
            A, b = assemble_system(J, -F, bcs_du)
            solve(A, du.vector(), b, tol=1e-12)

            #pylab.clf()
            #drawgrid = numpy.logspace(0.5,3,100)
            ##drawgrid -= drawgrid[0]
            #yvals = [du(r) for r in drawgrid]
            ##pylab.loglog(drawgrid,yvals,'bx-')
            #pylab.plot(drawgrid,yvals,'bx-')
            #print "CHECK",du(0),du(self.rs[-1])
            #pylab.title("dn")
            #gui.pause()

            mu = self.felr.integrate_scalar_field(du)
            du.vector()[:] = du.vector() - (mu/(self.felr.vol))
            print "Mu",mu
            
            #tot = sum(du.vector().array())
            eps = self.felr.integrate_scalar_field(abs(du))#/self.felr.vol
            print 'Iter',self.iters,'norm:', eps

            step = omega
            #if eps < 1000:
            #    tmp.vector()[:] = numpy.divide(u_k.vector(),du.vector())
            #    tmpvec = tmp.vector()
            #    maxstep = tmpvec[tmpvec>0.0].min()    
            #    maxstep = max(maxstep,1e-8)
            #
            #    if step > maxstep:
            #        step = maxstep
            
            u_.vector()[:] = u_k.vector() + step*du.vector()
            u_k.assign(u_)

            nvec = u_k.vector()
            negs =(nvec<=0.0).sum()
            minval = nvec.min()
            print "Number of negative density elements:",negs,"min value",minval
            #if minval <= 0.0:
            #    nvec[:] = nvec - minval 
            if negs > 0:
                minpos = nvec[nvec>0.0].min() 
                nvec[nvec<=0.0] = minpos

            u_k_pre.assign(u_k)

            intn = self.felr.integrate_scalar_field(u_k)
            print "Electrons before final correction:",intn
            u_k.vector()[:] = u_k.vector()*self.N/intn    
            intn = self.felr.integrate_scalar_field(u_k)
            print "Electrons after final correction:",intn

            pylab.clf()
            drawgrid = numpy.logspace(-5,2,100)
            #drawgrid -= drawgrid[0]
            yvals = [u_k(r) for r in drawgrid]
            yvals22 = [u_k_pre(r) for r in drawgrid]
            #yvals2 = [self.n(r) for r in drawgrid]
            pylab.loglog(drawgrid,yvals,'bx-')
            pylab.loglog(drawgrid,yvals22,'ro-')
            #pylab.loglog(drawgrid,yvals2,'k.-')
            pylab.title("n")
            pylab.pause(0.0001)
            #gui.pause()

            if eps < tol or self.iters > maxiter:
                break

        self.n = u_k



    def solve_newton_logn(self):
        
        fel = self.felr
        V = fel.V
        msh = fel.msh
        vol = fel.vol

        alpha = 4.0/1.0*(2.0*self.Z/(9.0*pi**2))**(1.0/3.0)
        
        def inner_boundary(x, on_boundary):
            return on_boundary and near(x[0],self.rs[0])
        
        def outer_boundary(x, on_boundary):
            return on_boundary and near(x[0],self.rs[-1])
        
        #u_k = Function(V)
        #self.prepare_starting_density()
        #u_k.assign(self.n)
        #u_k.vector()[:]=u_k.vector() + 0.1
        #n = self.felr.scalar_field_from_expr('rho0*(end-x[0])+100',rho0=0.5*self.Z**3,end=self.rs[-1]+(self.rs[-1]-self.rs[-2]))

        #mu = 0.09
        mu = 0.0
        
        n = interpolate(Constant(1.0),V)
        intn = self.felr.integrate_scalar_field(n)
        print "Density integrated before adjustment:",intn
        n.vector()[:] = n.vector()*self.N/intn  
        intn = self.felr.integrate_scalar_field(n)
        print "Density integrated after first adjustment:",intn                
        #u_k = self.felr.scalar_field_from_expr('rho0*pow(x[0],(-3.0/2.0))',Z=self.Z,fac=1.8,rho0=0.1*self.Z**3)
        u_k = project(ln(n),V)
        last_u_k = Function(V)

        pylab.clf()
        drawgrid = numpy.logspace(-3,1,100)
        yvals = [n(r) for r in drawgrid]
        pylab.plot(drawgrid,yvals,'bx-')
        pylab.title("n")
        gui.pause()

        
        #u_k = interpolate(Constant(self.Z/self.felr.vol), V)  # previous (known) u
        n_pre = Function(V)
        v_h = Function(V)
        rhs = Function(V)
        tmp = Function(V)
        
        u = TrialFunction(V)
        v = TestFunction(V)

        du_ = TrialFunction(V)

        du = Function(V)
        u_ = Function(V)  # u = u_k + omega*du'''''''
        omega = self.params['basestep']       # relaxation parameter
        eps = 1.0
        self.iters = 0
        maxiter = 2000
        while eps > self.params['eps'] and self.iters < maxiter:
            self.iters += 1

            rhs.assign(n)
            feradial.solve_radial_poisson(self.felr,v_h,rhs,Z=0,boundaryexpr=self.felr.expression("Z/x[0]",Z=self.Z))

            if self.params['cuspcond']:
                timer.start("Preparing cusp correction")
                n0 = n(0)
                if n0 <= 0.0:
                    k = Constant(sqrt(5.0/9.0*self.CF)*(0.5*self.Z**3)**(1.0/3.0))
                    potexpr = (-self.Z/r * (1 - exp(-2*k*r)) - k*self.Z*exp(-2*k*r))
                    print "== k (estimated):",sqrt(5.0/9.0*self.CF)*(0.5*self.Z**3)**(1.0/3.0)
                else:    
                    k = Constant(sqrt(5.0/9.0*self.CF)*n0**(1.0/3.0))
                    print "== k value used:",sqrt(5.0/9.0*self.CF)*n0**(1.0/3.0)
                    potexpr = self.felr.scalar_field_from_expr('(x[0] != 0.0)?(-Z/x[0] * (1.0 - exp(-2.0*k*x[0])) - k*Z*exp(-2.0*k*x[0])):(-3*k*Z)',k=k,Z=self.Z)                    
                timer.stop()                                
            else:   
                potexpr = -self.Z/r

            r = SpatialCoordinate(msh)[0]
            funcexpr = (5.0/3.0)*self.CF*exp(2.0/3.0*u_k)
            #funcexpr = (5.0/3.0)*self.CF*pow(n,2.0/3.0)

            F = (funcexpr + potexpr + v_h + Constant(mu))*v*dx

            #F = (math.log(((5.0/3.0)*self.CF)) + (2.0/3.0)*u_k - math.log(self.Z/r + v_h))*v*dx
            #F = ((5.0/3.0)*self.CF*pow(u_k,(2.0/3.0)) + t)*v*dx

            #bce = Constant(0.0)
            ##bce = 1.0/(self.rs[-1]**6)
            #bc0 = DirichletBC(V, bce, outer_boundary)
            #bcs_du = [bc0]
            bcs_du = []

            J = derivative(F, u_k, du_)
            A, b = assemble_system(J, -F, bcs_du)
            solve(A, du.vector(), b)

            #tot = sum(du.vector().array())
            #eps = numpy.linalg.norm(du.vector().array(), ord=numpy.Inf)
            #eps = self.felr.integrate_scalar_field(abs(dn - Constant(mu)))#/self.felr.vol
            #print 'Iter',self.iters,'norm:', eps

            #pylab.clf()
            #drawgrid = numpy.logspace(0.5,1,100)
            #drawgrid -= drawgrid[0]
            #yvals = [du(r) for r in drawgrid]
            #pylab.loglog(drawgrid,yvals,'bx-')
            #pylab.plot(drawgrid,yvals,'bx-')
            #pylab.title("du")
            #gui.pause()


            ################### TODO: Determine mu
            #mu = self.felr.integrate_scalar_field(du)
            #du.vector()[:] = du.vector() - (mu/(self.felr.vol))
            #print "Mu",mu
            #print "VAD?",omega
            step = omega
            #dn = exp(u_k + step*du) - exp(u_k)
            #mu = self.felr.integrate_scalar_field(dn)
            #n = project(n + dn - Constant(mu/self.felr.vol),V)
            #dn = exp(u_k + du) - exp(u_k)

            #dmu = self.felr.integrate_scalar_field(dn)/self.felr.vol
            #n = project(n + Constant(step)*(dn - Constant(dmu)),V)
            #dnv = project(dn - Constant(dmu),V)
            #mu += dmu
            #print "Mu is",mu

            u_.vector()[:] = u_k.vector() + step*du.vector()
            u_k.assign(u_)
            n.vector()[:] = numpy.exp(u_k.vector().array())
            #n = project(exp(u_k),V)
            #intn = self.felr.integrate_scalar_field(n)
            #print "Electrons before final correction:",intn

            #dmu = (self.N - intn)/self.felr.vol            
            #mu += 0.000*omega*dmu
            
            #fix = self.N/intn
            #n.vector()[:] = n.vector()*fix   
            #u_k.vector()[:] = u_k.vector() + ln(fix)
            #n.vector()[:] = numpy.exp(u_k.vector().array())
            #intn = self.felr.integrate_scalar_field(n)
            #print "Electrons after final correction:",intn

            #u_ = project(ln(n),V) 


            #eps = numpy.linalg.norm(u_k.vector().array() - last_u_k.vector().array(), ord=numpy.Inf)
            eps = numpy.linalg.norm(du.vector().array(), ord=numpy.Inf)
            print 'Iter',self.iters,'norm:', eps
            last_u_k.assign(u_k)
 
            intn = self.felr.integrate_scalar_field(n)
            print "Electrons in system:",intn

            #if eps < 1000:
            #    tmp.vector()[:] = numpy.divide(u_k.vector(),du.vector())
            #    tmpvec = tmp.vector()
            #    maxstep = tmpvec[tmpvec>0.0].min()    
            #    maxstep = max(maxstep,1e-8)
            #
            #    if step > maxstep:
            #        step = maxstep
            
            #pylab.clf()
            #drawgrid = numpy.logspace(0.5,2,100)
            ##drawgrid -= drawgrid[0]
            #print "CHECK u_k",u_k(0),u_k(self.rs[-1])
            #yvals = [u_k(r) for r in drawgrid]
            ##pylab.loglog(drawgrid,yvals,'bx-')
            #pylab.plot(drawgrid,yvals,'bx-')
            #pylab.title("u_k")
            #gui.pause()

            #nvec = u_k.vector()
            #nvec[nvec>100] = 10

            #n = project(exp(u_k),V)
            #n.vector()[:] = numpy.exp(u_k.vector())

            #nvec = u_k.vector()
            #negs =(nvec<=0.0).sum()
            #minval = nvec.min()
            #print "Number of negative density elements:",negs,"min value",minval
            #if minval <= 0.0:
            #    nvec[:] = nvec - minval 
            #if negs > 0:
            #    minpos = nvec[nvec>0.0].min() 
            #    nvec[nvec<=0.0] = minpos

            n_pre.assign(n)

            #intn = self.felr.integrate_scalar_field(n)
            #print "Electrons before final correction:",intn
            #n.vector()[:] = n.vector()*self.N/intn    
            #intn = self.felr.integrate_scalar_field(n)
            #print "Electrons after final correction:",intn

            pylab.clf()
            drawgrid = numpy.logspace(-3,1,100)
            #drawgrid -= drawgrid[0]
            yvals = [n(r) for r in drawgrid]
            yvals3 = [self.n(r) for r in drawgrid]
            yvals22 = [n_pre(r) for r in drawgrid]
            #yvals2 = [self.n(r) for r in drawgrid]
            pylab.loglog(drawgrid,yvals,'bx-')
            pylab.loglog(drawgrid,yvals22,'ro-')
            pylab.loglog(drawgrid,yvals3,'k.-')
            #pylab.loglog(drawgrid,yvals2,'k.-')
            pylab.title("n")
            pylab.pause(0.0001)
            pylab.ioff()
            pylab.show()

        self.n = n




    def solve_decsent_n(self):

        fel = self.felr
        V = fel.V
        msh = fel.msh
        vol = fel.vol

        alpha = 4.0/1.0*(2.0*self.Z/(9.0*pi**2))**(1.0/3.0)
        
        def inner_boundary(x, on_boundary):
            return on_boundary and near(x[0],self.rs[0])
        
        def outer_boundary(x, on_boundary):
            return on_boundary and near(x[0],self.rs[-1])
        
        #u_k = Function(V)
        #self.prepare_starting_density()
        #u_k.assign(self.n)
        #u_k.vector()[:]=u_k.vector() + 0.1
        u_k = self.felr.scalar_field_from_expr('rho0*(end-x[0])',rho0=0.5*self.Z**3,end=self.rs[-1])
        #u_k = self.felr.scalar_field_from_expr('rho0*pow(x[0],(-3.0/2.0))',Z=self.Z,fac=1.8,rho0=0.1*self.Z**3)
        intn = self.felr.integrate_scalar_field(u_k)
        print "Density integrated before adjustment:",intn
        u_k.vector()[:] = u_k.vector()*self.N/intn  
        intn = self.felr.integrate_scalar_field(u_k)
        print "Density integrated after first adjustment:",intn        
        
        #u_k = interpolate(Constant(self.Z/self.felr.vol), V)  # previous (known) u
        u_k_pre = Function(V)
        v_h = Function(V)
        rhs = Function(V)
        tmp = Function(V)

        u = TrialFunction(V)
        v = TestFunction(V)

        du_ = TrialFunction(V)

        du = Function(V)
        u_ = Function(V)  # u = u_k + omega*du
        omega = self.params['basestep']        # relaxation parameter
        eps = 1.0
        tol = 1.0E-6
        self.iters = 0
        maxiter = 200
        r = SpatialCoordinate(msh)[0]
        while eps > tol and self.iters < maxiter:
            self.iters += 1

            rhs.assign(u_k)
            feradial.solve_radial_poisson(self.felr,v_h,rhs,Z=0,boundaryexpr=self.felr.expression("Z/x[0]",Z=self.Z))

            if self.params['cuspcond']:
                timer.start("Preparing cusp correction")
                n0 = u_k(0)
                if n0 <= 0.0:
                    k = Constant(sqrt(5.0/9.0*self.CF)*(0.5*self.Z**3)**(1.0/3.0))
                    potexpr = (-self.Z/r * (1 - exp(-2*k*r)) - k*self.Z*exp(-2*k*r))
                    print "== k (estimated):",sqrt(5.0/9.0*self.CF)*(0.5*self.Z**3)**(1.0/3.0)
                else:    
                    k = Constant(sqrt(5.0/9.0*self.CF)*n0**(1.0/3.0))
                    print "== k value used:",sqrt(5.0/9.0*self.CF)*n0**(1.0/3.0)
                    potexpr = self.felr.scalar_field_from_expr('(x[0] != 0.0)?(-Z/x[0] * (1.0 - exp(-2.0*k*x[0])) - k*Z*exp(-2.0*k*x[0])):(-3*k*Z)',k=k,Z=self.Z)
                    
                timer.stop()                                
            else:   
                potexpr = -self.Z/r


            #nvec = u_k.vector()
            #negs =(nvec<0.0).sum()
            #minval = nvec.min()
            #print "** Number of negative density elements:",negs,"min value",minval

            # Why is this needed?!
            #t = Function(V)
            #t = project(v_h - self.Z/r,V)

            #pylab.clf()
            #drawgrid = numpy.logspace(0.1,1.2,100)
            #drawgrid -= drawgrid[0]
            #yvals = [-t(r) for r in drawgrid]
            #yvals2 = [v_h(r) for r in drawgrid]
            #pylab.loglog(drawgrid,yvals,'bx-')
            #pylab.loglog(drawgrid,yvals2,'ro-')
            #pylab.title("v_H")
            #gui.pause()

            print "Calculating dn"
            du = project((5.0/3.0)*self.CF*pow(u_k,(2.0/3.0)) + v_h + potexpr,V)
            print "Finished"
            #F = ((5.0/3.0)*self.CF*pow(u_k,(2.0/3.0)) + t)*v*dx

            #bce = Constant(0.0)
            ##bce = 1.0/(self.rs[-1]**6)
            #bc0 = DirichletBC(V, bce, outer_boundary)
            #bcs_du = [bc0]
            #bcs_du = []

            #J = derivative(F, u_k, du_)
            #A, b = assemble_system(J, -F, bcs_du)
            #solve(A, du.vector(), b)

            #pylab.clf()
            #drawgrid = numpy.logspace(0.5,3,100)
            ###drawgrid -= drawgrid[0]
            #yvals = [du(r) for r in drawgrid]
            ##pylab.loglog(drawgrid,yvals,'bx-')
            #pylab.plot(drawgrid,yvals,'bx-')
            #print "CHECK",du(0),du(self.rs[-1])
            #pylab.title("dn")
            #gui.pause()

            mu = self.felr.integrate_scalar_field(du)
            du.vector()[:] = du.vector() - (mu/(self.felr.vol))
            print "Mu",mu
            
            #tot = sum(du.vector().array())
            eps = self.felr.integrate_scalar_field(abs(du))#/self.felr.vol
            print 'Iter',self.iters,'norm:', eps

            step = omega
            #if eps < 1000:
            #    tmp.vector()[:] = numpy.divide(u_k.vector(),du.vector())
            #    tmpvec = tmp.vector()
            #    maxstep = tmpvec[tmpvec>0.0].min()    
            #    maxstep = max(maxstep,1e-8)
            #
            #    if step > maxstep:
            #        step = maxstep
            
            u_.vector()[:] = u_k.vector() - step*du.vector()
            u_k.assign(u_)

            nvec = u_k.vector()
            negs =(nvec<=0.0).sum()
            minval = nvec.min()
            print "Number of negative density elements:",negs,"min value",minval
            #if minval <= 0.0:
            #    nvec[:] = nvec - minval 
            if negs > 0:
                nvec[nvec<=1e-12] = 1e-12

            u_k_pre.assign(u_k)

            intn = self.felr.integrate_scalar_field(u_k)
            print "Electrons before final correction:",intn
            u_k.vector()[:] = u_k.vector()*self.N/intn    
            intn = self.felr.integrate_scalar_field(u_k)
            print "Electrons after final correction:",intn

            pylab.clf()
            drawgrid = numpy.logspace(-5,2,100)
            #drawgrid -= drawgrid[0]
            yvals = [u_k(rr) for rr in drawgrid]
            yvals22 = [u_k_pre(rr) for rr in drawgrid]
            #yvals2 = [self.n(r) for r in drawgrid]
            pylab.loglog(drawgrid,yvals,'bx-')
            pylab.loglog(drawgrid,yvals22,'ro-')
            #pylab.loglog(drawgrid,yvals2,'k.-')
            pylab.title("n")
            pylab.pause(0.0001)
            #gui.pause()

            if eps < tol or self.iters > maxiter:
                break

        self.n = u_k


            
    def solve(self):
        startstep = self.params['basestep']

        #intn = self.felr.integrate_scalar_field(self.n)
        #print "Number of electrons in starting density:",intn
        #drawgrid = numpy.r_[0:10:0.1]
        #gui.plot_radial(drawgrid,self.n,"Starting density")
        #gui.pause()

        while True:        
            
            self.prepare_starting_density()
            
            basestep = startstep
            restart_all = False
            absdns = []
            trouble_steps = 0
            troublefree_steps = 0
            increased_last_step = False
            backtrack_step = False
                    
            n = self.n
            prevn = self.felr.new_scalar_field()
            prevn.assign(n)
            precorrn = self.felr.new_scalar_field()
            precorrn.assign(n)
            
            self.tfn = self.felr.new_scalar_field()
            dn = self.felr.new_scalar_field()
            rhs = self.felr.new_scalar_field()
            tmp = self.felr.new_scalar_field()
            tmp2 = self.felr.new_scalar_field()
            tmp3 = self.felr.new_scalar_field()
            self.cusppot = None
            functional = self.felr.new_scalar_field()

            timer.start("Solve unmodified TF equation")
            feradial.solve_radial_tf(self.felr, self.tfn, self.Z)
            timer.stop()
    
            timer.start("Total solver time")
    
            timer.start("Computing gradients and laplacian")
            gradn, lapln = self.felr.grad_lapl(n)
            timer.stop()
    
            lasterr = None
            self.iters=0
            converged = False
            while not converged:    
                print "************ TROUBLE_STEPS",trouble_steps
    
                self.iters+=1
                print "==== Iteration:",self.iters
                        
                #plot1d(lapln,"lapln")
            
                timer.start("Computing ked functional")
                self.functional_potential(functional)        
                timer.stop()
                
                # J. Chem,. Phys 124, 124107 (2006)
                #if cuspcond:
                #    print "== Calculating cusp condition contribution"
                #    tmp = interpolate(Zero(element = V.ufl_element()),V)
                #    for i in range(len(atoms)):
                #        rhs.vector()[:] = 0.0
                #        vext.vector()[:] = 0.0
                #        solve_poisson(vext,rhs, [deltas[i]])
                #        k = sqrt(5.0/9.0*CF)*abs(n(cartcoords[i]))**(1.0/3.0)
                #        print "= Interpolating cusp condition expression"
                #        cuspexp = interpolate(Cusppot(k,element = V.ufl_element()),V)
                #        print "= Finished"
                #        tmp.vector()[:] = tmp.vector() + vext.vector()*(1.0-cuspexp.vector()) - atoms[i]*k*cuspexp.vector()
                #    dn.vector()[:] = dn.vector() + tmp.vector() 
                #    print "== Finished"
                if self.params['cuspcond']:
                    timer.start("Computing hartree potential")
                    rhs.vector()[:] = self.n.vector() 
                    dn.vector()[:] = 0.0 
                    #feradial.solve_radial_poisson(self.felr, dn, rhs)
                    feradial.solve_radial_poisson(self.felr, dn, rhs, boundaryexpr=self.felr.expression("Z/x[0]",Z=self.Z))
    
                    timer.stop()
                    
                    #tmp = self.felr.scalar_field_from_expr('(x[0] > 1.0)?(Z/x[0]):(Z)',Z=self.Z)
                    #drawgrid = numpy.logspace(-5,1,100)
                    #gui.plot_radial(drawgrid,[dn,tmp],"dn")
                    #gui.pause()
                    
                    timer.start("Preparing cusp correction")
                    #print n.vector().max(),n(0)
                    #gui.pause()
                    k = sqrt(5.0/9.0*self.CF)*n(0)**(1.0/3.0)
                    #cuspexp = self.felr.scalar_field_from_expr('(x[0] != 0.0)?(1.0):(0.0)',k=k,Z=self.Z)
                    cuspexp = self.felr.scalar_field_from_expr('(x[0] != 0.0)?(-Z/x[0] * (1.0 - exp(-2.0*k*x[0])) - k*Z*exp(-2.0*k*x[0])):(-k*Z-2.0*k*Z)',k=k,Z=self.Z)
    
                    #tmp2.vector()[:] = cuspexp.vector() + dn.vector() 
                    #tmp = self.felr.scalar_field_from_expr('(x[0] > 1.0)?(-Z/x[0]):(-Z)',Z=self.Z)
                    #drawgrid = numpy.logspace(-5,1,100)
                    #gui.plot_radial(drawgrid,[dn,cuspexp,tmp,tmp2,functional],"dn")
                    #gui.pause()
        
                    dn.vector()[:] = dn.vector() + cuspexp.vector() + functional.vector() 
    
                    #drawgrid = numpy.logspace(-5,1,100)
                    #gui.plot_radial(drawgrid,[dn],"dn")
                    #gui.pause()
    
                    
                    timer.stop() #Computing cusp condition potential"
                else:
                    timer.start("Computing hartree + external potential")
                    rhs.vector()[:] = n.vector()    
                    feradial.solve_radial_poisson(self.felr, dn, rhs, self.Z)
                    dn.vector()[:] = dn.vector() + functional.vector() 
                    timer.stop()
            
                #plot1d([functional,n],"dn")
                        
                #plot1d(vh,"vh")
                #plot1d(tfpot,"ftpot")
                #plot1d(vext,"vext")    
                #plot1d(cusppot,"cusppot")    
                #plot1d(dn,"dn")    
            
                timer.start("Computing chemical potential")
                mu = self.felr.integrate_scalar_field(dn)
                dn.vector()[:] = dn.vector() - (mu/(self.felr.vol))
    
                # More advanced iteration to make sure to minimize numerical error            
                #mu = 0.0
                #tmp.assign(dn)
                #intn = self.felr.integrate_scalar_field(tmp)
                #previntn = intn
                #while abs(intn) > 1e-12:
                #    tmp.vector()[:] = tmp.vector() - (intn/(self.felr.vol))
                #    intn = self.felr.integrate_scalar_field(tmp)
                #    if abs(intn) >= abs(previntn):
                #        print "final my-step:",intn,mu
                #        break 
                #    dn.assign(tmp)
                #    mu += previntn
                #    print "my-step:",intn,mu
                #    previntn = intn
    
                print "MU:",mu,"(adj:",mu/self.felr.vol,")"
    
                # Second adjustment
                #intn = self.felr.integrate_scalar_field(dn)
                #dn.vector()[:] = dn.vector() - intn/self.felr.vol
                #print "************ Integrated density step:",intn
    
                #intn = self.felr.integrate_scalar_field(dn)
                #print "************ ## Integrated density step:",intn,mu - (mu/self.felr.vol)*self.felr.vol
    
                #tmp = self.felr.scalar_field_from_expr("1.0")
                #intn = self.felr.integrate_scalar_field(tmp)
                #print "************ ## Realvol:",intn,self.felr.vol
    
                #drawgrid = numpy.r_[0:10:0.1]
                #drawgrid = numpy.logspace(-5,1,100)
                #gui.plot_radial(drawgrid,dn,"dn")
                #gui.pause()
                
                timer.stop()
                
                timer.start("Computing |dn|")
                err = self.felr.integrate_scalar_field(abs(dn))#/self.felr.vol
                absdns += [err] 
                timer.stop()
    
                print "|dn|=",err
                if err < self.params['eps']:
                    converged = True
    #             if len(absdns)>1 and err > absdns[-1] and err > absdns[-2]:
    #                 basestep = basestep / 1.5
    #             elif len(absdns)>0 and err > absdns[-1]:
    #                 basestep = basestep / 1.1
    #             elif len(absdns)>3 and err < absdns[-1] and err < absdns[-2] and err < absdns[-3]:
    #                 basestep = basestep * 1.1
                #print "CHECK",err,absdns[-2]
                #gui.pause()
                
                # Overengineered step control should force convergence in almost any situation
                if (len(absdns)>1 and err > absdns[-2]):
                    if len(absdns) <= 3:
                        startstep = startstep / 2.0
                        restart_all = True
                        break 
                    trouble_steps += 1
                    troublefree_steps = 0
                    increased_last_step = False
                    if trouble_steps < 5:
                        del absdns[-1]
                        n.assign(prevn)
                        basestep = basestep / 1.1
                        backtrack_step = True
                        print "==== Last step goes away from solution, jump back with smaller step length"
                        continue
                    else:
                        print "GURK",len(absdns)
                        if len(absdns) <= 10:
                            startstep = startstep / 2.0
                            restart_all = True
                            break 
                        # If we appear completely stuck, revert the step length and see if we can "kick" things back to convergence
                        basestep = basestep * 1.1 * 1.1 * 1.1 * 1.1 * 1.1
                        trouble_steps = 0                    
                #else:
                #    gui.pause()
                elif troublefree_steps > 5 and not backtrack_step:
                    trouble_steps = 0                
                    troublefree_steps += 1
                    if increased_last_step:
                        if (absdns[-3] - absdns[-2]) > (absdns[-2] - err):
                            basestep = basestep / 1.1
                            #print absdns[-3], absdns[-2], err, "###",(absdns[-3] - absdns[-2]), (absdns[-2] - err)
                            #gui.pause()
                        increased_last_step = False
                    else:
                        basestep = basestep * 1.1
                        increased_last_step = True
                else:
                    if not backtrack_step:
                        trouble_steps = 0                
                        troublefree_steps += 1
                        increased_last_step = False    
    
                if backtrack_step:
                    backtrack_step = False
    
                step = basestep
                #if err < 0.1:
                #step = basestep*err**(1.0/16.0) 
                #if err < 1.0:
                #    step = basestep*err**(1.0/25.0) 
                #else:
                #    step = basestep
                #step = basestep*err**(1.0/32.0) 
                print "== Step length: ",step
                lasterr = err
                        
                # Maxstep algorithm has very bad convergence rate
                #tmp2.vector()[:] = numpy.divide(n.vector(),dn.vector())
                #tmp2vec = tmp2.vector()
                #maxstep = tmp2vec[tmp2vec>0.0].min()    
                #maxstep = max(maxstep,1e-8)
                #print "******************************MAXSTEP:",maxstep,step,maxstep < step
                #if maxstep < step:
                #    step = maxstep
                    
                #intn = self.felr.integrate_scalar_field(dn)
                #print "************ Integrated density step:",intn
                #intn = self.felr.integrate_scalar_field(n)
                #print "************ Integrated density before:",intn
    
                timer.start("Taking density step")
                prevn.assign(n)
                # DEBUG TEST
                n.vector()[:] = n.vector() - step * dn.vector()
                #n.vector()[:] = n.vector() - step * 2.0/pi*numpy.arctan(n.vector()*1e12) * dn.vector()
                precorrn.assign(n)
                
                #intn = self.felr.integrate_scalar_field(tmp)
                #previntn = intn
                #while abs(intn)-self.N > 1e-12:
                #    tmp2.vector()[:] = tmp2.vector() - ((intn - self.N)/(self.felr.vol))
                #    intn = self.felr.integrate_scalar_field(tmp2)
                #    if abs(intn)-self.N >= abs(previntn)-self.N:
                #        print "final dens adj step:",intn,previntn
                #        break 
                #    tmp.assign(tmp2)
                #    print "dens adj step:",intn,mu
                #    previntn = intn
                #
                #intn = self.felr.integrate_scalar_field(tmp)
                #print "************ Integrated density afrer:",intn
                            
                timer.start("Computing gradients and laplacian")
                gradn, lapln = self.felr.grad_lapl(n)
                timer.stop()
            
                #print "== Prepare comparing density"
                #i = 0
                #compare = Expression('rho0*exp(-2.0*Z*sqrt((x[0]-ax)*(x[0]-ax) + (x[1]-ay)*(x[1]-ay) + (x[2]-az)*(x[2]-az)))',Z=atoms[i],ax=cartcoords[i][0],ay=cartcoords[i][1],az=cartcoords[i][2],rho0=tmp(cartcoords[i]),element = V.ufl_element())
                #compare = interpolate(compare,V)
                #print "== Finished"
                timer.start("Calculating energies")
                self.calculate_energies()
                data = ['i:%d'%(self.iters,),'$|dn|$: %.4e'%(err,),'step: %.4e'%(step,),'bad steps: %d'%(trouble_steps,),'','$n(max)$:%.4e'%(n([self.rs[-1]]),),'$n(r_1)/Z_1^3$:%.4f'%(n([0])/self.Z**3,),
                        '$-E/Z_1^{7/3}$:%.4f'%(-(self.e_ionelec+self.e_elecelec+self.e_functional)/self.Z**(7.0/3.0),)]
                if self.params['evunits']:
                    convfact = self.hartree2ev           
                else:
                    convfact = 1.0
                data += ['','$E_{ii}$: %.4f'%(self.e_ionion*convfact,),'$E_{ee}$: %.4f'%(self.e_elecelec*convfact,),
                         '$E_{ie}$: %.4f'%(self.e_ionelec*convfact,),'$E_{f}$: %.4f'%(self.e_functional*convfact,),
                         '','$E_{tot}$: %.4f'%(self.e_tot*convfact,)]
                timer.stop()
                            
                if self.params['dynrefine'] and not converged: #iters%5 == 0:
                    timer.start("Dynamically refine grid")
                    self.fe.refine_mesh(n, gradn, lapln)
                    timer.stop()
                    timer.start("Transfer solution")
                    n = self.fe.transfer_scalar_field(n)
                    timer.stop()
                    timer.start("Transfer grad and lapl, re-allocate other fields")
                    gradn = self.fe.transfer_scalar_field(gradn)
                    lapln = self.fe.transfer_scalar_field(lapln)
    
                    prevn = self.fe.transfer_scalar_field(prevn)
                    dn = Function(self.V)
                    rhs = Function(self.V)
                    vext = Function(self.V)
                    functional = Function(self.V)
                    cusppot = None
                    
                    timer.stop()
    
                intn = self.felr.integrate_scalar_field(n)
                print "Electrons before any adjustment:",intn
    
                timer.start("Fixing negative density and correct for lost/added electrons")
                nvec = n.vector()
                negs =(nvec<0.0).sum()
                minval = nvec.min()
                print "Number of negative density elements:",negs,"min value",minval
                if minval < 0.0:
                    nvec[:] = nvec - minval
#                 if negs > 0:
#                     nvec[nvec<0] = 0.0
#     
#                     tmp3.assign(n)
#                     nvec = tmp3.vector()
#                     intn = self.felr.integrate_scalar_field(tmp3)
#                     previntn = intn
#                     while intn > self.N:
#                         minval = nvec[nvec>0].min()
#                         nvec[nvec<=minval]=0.0
#                         intn = self.felr.integrate_scalar_field(tmp3)
#                         if abs(intn)-self.N >= abs(previntn)-self.N:
#                             print "final dens adj step:",intn,previntn
#                             break 
#                         n.assign(tmp3)
#                         print "dens adj step:",intn
#                         previntn = intn
            
                intn = self.felr.integrate_scalar_field(n)
                print "Electrons before final correction:",intn
                n.vector()[:] = n.vector()*self.N/intn    
                intn = self.felr.integrate_scalar_field(n)
                print "Electrons after final correction:",intn
                #if abs(intn - self.N) > 1e-6: 
                #    minval = n.vector().min()
                #    adjustment = (intn - self.N)/self.felr.vol
                #    if adjustment < minval and minval > 0.0:
                #        n.vector()[:] = n.vector() - adjustment
                #    else:
                #        n.vector()[:] = n.vector()*self.N/intn    
                #
                #    intn = self.felr.integrate_scalar_field(n)
                #    print "Electrons after final correction:",intn
                    
                timer.stop()
    
                timer.start("Plotting")
                #drawgrid = numpy.r_[0:10:0.1]
                drawgrid = numpy.logspace(-5,1,100)
                gui.plot_radial(drawgrid,[n,prevn,precorrn],"density",data,xs=drawgrid)
                #gui.pause()
                timer.stop()
                        
                print "==== Iteration finished"
                
                #nvec = n.vector()
                #negs =(nvec<0.0).sum()
                #print "== Negative density elements after everything is finished:",(nvec<0.0).sum()
            if not restart_all:
                break

        print "Reached convergence after",self.iters,"iterations"
        timer.stop()
    
    def calculate_energies(self):    

        timer.start("Calculate ion-ion energy")
        field = self.felr.new_scalar_field()
        tmp = self.felr.new_scalar_field()
        rhs = self.felr.new_scalar_field()

        self.e_ionion = 0.0    
        timer.stop()
        
        timer.start("Calculate ion-electron energy")
        
        rhs.assign(self.n)
        field.vector()[:]=0.0
        # TODO: better to solve *with* delta and adjust with 1/r?
        feradial.solve_radial_poisson(self.felr, field, rhs, Z=0,boundaryexpr=self.felr.expression("Z/x[0]",Z=self.Z))
        #self.e_ionelec = -self.Z*field([0.0])
        #self.e_ionelec = self.felr.integrate_scalar_field(self.felr.expression("-Z/x[0]",Z=self.Z)*self.n)
        r = SpatialCoordinate(self.felr.msh)[0]
        self.e_ionelec = self.felr.integrate_scalar_field(-self.Z/r*self.n)
    
        timer.stop()

        timer.start("Calculate classical elec-elec energy (hartree term)")
        
        #rhs.assign(self.n)
        #field.vector()[:]=0.0
        #fe.solve_poisson(self.fe, field, rhs)
           
        self.e_elecelec = 0.5*self.felr.integrate_scalar_field(field*self.n)
    
        timer.stop()
        
        
        timer.start("Calculate functional energy")
        
        #self.functional_energy_density(field)
        #self.e_functional = self.felr.integrate_scalar_field(field)
    
        self.e_functional = self.felr.integrate_scalar_field(self.functional_energy_density_expr())
    
        timer.stop()

        self.e_tot = self.e_ionion + self.e_ionelec + self.e_elecelec + self.e_functional

    def print_energies(self):            
        print "Solution found in",self.iters,"iterations."
        if self.params['evunits']:
            convfact =      self.hartree2ev           
            print "==== Resulting energies (eV): ================"
        else:
            convfact = 1.0
            print "==== Resulting energies (hartree): ==========="
        print "Ion-ion:        % 10.4f"%(self.e_ionion*convfact,)
        print "Ion-elec:       % 10.4f"%(self.e_ionelec*convfact,)
        print "Elec-elec (Eh): % 10.4f"%(self.e_elecelec*convfact,)
        print "Functional:     % 10.4f"%(self.e_functional*convfact,)
        print "=============================================="
        print "Total energy:   % 10.4f"%(self.e_tot*convfact,)
        print "=============================================="
        print
        #print "==== Some other data:"
        #print "n(r_1)/Z_1^3:",self.n([0.0])/self.Z**3
        #print "-E/Z_1^(7/3):",-(self.e_ionelec+self.e_elecelec+self.e_functional)/self.Z**(7.0/3.0)
        #print "-T/E:",(-self.e_functional)/(self.e_tot)
        #print "-E/E_TF:",(self.e_ionelec+self.e_elecelec+self.e_functional)/(-0.768745*self.Z**(7.0/3.0)),"(",(-0.768745*self.Z**(7.0/3.0))*convfact,":",((-0.768745*self.Z**(7.0/3.0))-self.e_tot)/(-0.768745*self.Z**(7.0/3.0)),")"
        #print "===="
        #print
        intn = self.felr.integrate_scalar_field(self.n)
        print "Number of electrons:",intn
        #drawgrid = numpy.r_[0:10:0.1]
        #gui.plot_radial(drawgrid,self.n)
        #gui.pause()
        