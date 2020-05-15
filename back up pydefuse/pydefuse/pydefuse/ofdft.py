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
import numpy
import pylab
from pydefuse.chronicle import chronicle 
from dolfin import *
from .densityobj import DensityFields, DensityWeakForm
from .functional import func_tf

from pydefuse import gui
from pydefuse import fe
from pydefuse.functional.weizsacker import func_weizsacker

class Ofdft(object):

    CF=3.0/10.0*(3.0*math.pi**2)**(2.0/3.0)
    CX=3.0/4.0*(3.0/math.pi)**(1.0/3.0)
    hartree2ev=27.21138386
    bohr2ang=.529177210818

    def __init__(self, unitcell, atoms, coords, N, pbc, functionals, **params):
        self.unitcell = unitcell
        self.coords = coords
        self.atoms = atoms
        self.pbc = pbc
        self.functionals = functionals
        self.data = None
        self.e_ionion = None
        
        if N == None:
            self.N = sum(atoms)
        else:
            self.N = N 
        self.cusppot = None
    
        # Defaults
        all_params = { 
          'basestep':1.0,
          'prerefine': False,
          'prerefinepower': 2.0,
          'dynrefine':False,
          'threadpar':False,
          'cuspcond':False,
          'eps':1e-6,
          'gui':True,
          'mesh_type':'structured_rectangle',
          'mesh_rectangle_divisions':6,
          'neg_density_fix':'maxstep',
          'start_density':'uniform',
          'element_order':1,
          'units':'ev',
          'loglevel':1,
          'ionion':'external'
        }
        all_params.update(params)

        # Move parameters dict into attributes for nicer syntax 
        class Attrdata: pass
        self.params = Attrdata()
        for param in all_params:
            setattr(self.params,param,all_params[param])

        if self.params.units == 'ev':
            self.convfact_energy = self.hartree2ev           
            self.convfact_energy_name = "eV"           
            self.convfact_length = self.bohr2ang           
            self.convfact_length_name = u"Å"           
        elif self.params.units == 'hartree':
            self.convfact_energy = 1.0
            self.convfact_energy_name = "ha"           
            self.convfact_length = 1.0
            self.convfact_length_name = "bohr"           
        elif self.params.units == 'rydberg':
            self.convfact_energy = 2.0
            self.convfact_energy_name = "ry"           
            self.convfact_length = 1.0
            self.convfact_length_name = "bohr"           

        chronicle.loglevel = self.params.loglevel

    def run(self):
        """
        Setup and solve the of-dft equations
        """
        chronicle.start("Total run time",silent=True)

        self.initialize()
        self.prepare_starting_mesh()
        self.prepare_starting_state()
        self.run_solver_newton_vh()
        self.calculate_energies()
        self.exit()

        chronicle.stop()
                
    def initialize(self):
        chronicle.start("Initialize system")
        self.fel = fe.FeLattice(self.unitcell, self.pbc, self.params)
        self.cartcoords = self.fel.reduced_to_cartesian(self.coords)
        self.deltastrengths = [-x for x in self.atoms]
        
        convfact = self.convfact_length
        print ("=====================================================")
        print (" SYSTEM:")
        print ("=====================================================")
        print ("Unitcell ("+self.convfact_length_name+"):")
        print (self.unitcell*convfact)
        print ("Periodic boundary conditions:",self.pbc)
        print ("Atoms:", self.atoms)
        print ("Coords ("+self.convfact_length_name+"):")
        print (self.coords*convfact)
        print ("Cell volume ("+self.convfact_length_name+"^3):",self.fel.vol*convfact**3)
        print ("Number of electrons in system:",self.N)
        print ("=====================================================")

        if self.params.gui:
            import pydefuse.gui
            gui.gui_init()
                
        chronicle.stop()

    def prepare_starting_mesh(self):
        chronicle.start("Prepare mesh")
        self.fel.create_mesh(self.cartcoords,self.deltastrengths)
        #plot(mesh,interactive = True)        
        chronicle.stop()
            
    def prepare_starting_state(self):
        chronicle.start("Prepare starting state")

        n = self.fel.new_scalar_field()        
        if self.params.start_density == 'uniform':
            n = interpolate(Constant(self.N/self.fel.vol), self.fel.V)
        elif self.params.start_density == 'exp':
            cartcoords = self.cartcoords
            startingdensity = 0
            for i in range(len(self.atoms)):
                startingdensity += self.fel.scalar_field_from_expr('rho0*exp(-2.0*Z*sqrt((x[0]-ax)*(x[0]-ax) + (x[1]-ay)*(x[1]-ay) + (x[2]-az)*(x[2]-az)))',
                                              Z=self.atoms[i],ax=cartcoords[i][0],ay=cartcoords[i][1],az=cartcoords[i][2],rho0=0.5*self.atoms[i]**3)
            n = interpolate(startingdensity,self.fel.V)
            intn = float(assemble((n)*dx(self.fel.msh)))
            chronicle.message("Density integrated before adjustment:"+str(intn))
            n.vector()[:] = n.vector()*self.N/intn  
            intn = float(assemble((n)*dx(self.fel.msh)))
            chronicle.message("Density integrated after adjustment:",str+(intn))
        else:
            raise Exception("Unknown value for start_density parameter.")
            
        #TODO: Is this questionable to apply regardless of functionals?
        if self.pbc != [True, True, True]:
            self.boundaryexpr = 0
            for i in range(len(self.atoms)):
                self.boundaryexpr += self.fel.expression("Z/sqrt((x[0]-ax)*(x[0]-ax) + (x[1]-ay)*(x[1]-ay) + (x[2]-az)*(x[2]-az))",
                                         Z=self.atoms[i],ax=self.cartcoords[i][0],ay=self.cartcoords[i][1],az=self.cartcoords[i][2])                
        else:
            self.boundaryexpr = None

        vh = interpolate(Constant(-1.0), self.fel.V)

        self.n = n
        self.vh = vh

        chronicle.stop()
                        
    def run_solver_newton_vh(self):
        chronicle.start("Solve the system using newton solver")

        if self.params.gui:
            import pydefuse.gui

        if hasattr(self.fel,'vext'):
            self.vext = self.fel.vext
        else:
            self.vext = None

        fel = self.fel
        V = fel.V
        W = fel.W
        cartcoords = self.cartcoords

        fake_mu = 0.0
        
        mixed_test_functions = TestFunction(W)
        (vr,pr) = split(mixed_test_functions)
          
        du_trial = TrialFunction(W)        
        du = Function(W)

        #vh = self.vh        
        #dvh = Function(V)
        #dn = Function(V)
        nlast = Function(V)

        # Initial state 
        n = self.n
        vh = self.vh
        u_k = self.fel.new_mixed_scalar_field()
        assign(u_k.sub(0),vh) 
        assign(u_k.sub(1),n)  

        potexpr = None

        bcs_du = []

        rhs = self.fel.new_scalar_field()
        eps = 1.0
        self.iters = 0
        maxiter = 5000

        if self.params.cuspcond:
            potexpr = self.fel.new_scalar_field()
        else:
            potexpr = self.vext
        
        while eps > self.params.eps and self.iters < maxiter:
            self.iters += 1

            if self.params.cuspcond:

                chronicle.start("Preparing cusp correction")

                if self.vext == None:
                    self.vext = self.fel.new_scalar_field()
                    rhs.vector()[:] = 0.0
                    fe.solve_poisson(self.fel, self.vext, rhs, self.cartcoords, self.deltastrengths)

                potexpr.vector()[:] = 0.0
                for i in range(len(self.atoms)):   
                    Z = self.atoms[i]
                    chronicle.message("Calculate cusp expontential for atom",i,"(",Z,")")
                    n0 = self.n(self.cartcoords[i]) 
                    if n0 <= 0.0:
                        k = Constant(sqrt(5.0/9.0*self.CF)*(0.5*Z**3)**(1.0/3.0))
                        #potexpr = (self.cusppot * (1 - exp(-2*k*r)) - k*Z*exp(-2*k*r))
                        chronicle.message("== k value used (estimated):",sqrt(5.0/9.0*self.CF)*(0.5*Z**3)**(1.0/3.0))
                    else:    
                        k = Constant(sqrt(5.0/9.0*self.CF)*n0**(1.0/3.0))
                        chronicle.message("== k value used:",sqrt(5.0/9.0*self.CF)*n0**(1.0/3.0))
                    #newfield = self.fel.scalar_field_from_expr('(x[0] != 0.0)?(-Z/x[0] * (1.0 - exp(-2.0*k*x[0])) - k*Z*exp(-2.0*k*x[0])):(-3*k*Z)',k=k,Z=Z)
                    cuspexp = self.fel.scalar_field_from_expr('(!(x[0]==ax && x[1]==ay && x[2]==az))?vext*(1.0-exp(-2.0*k*sqrt((x[0]-ax)*(x[0]-ax) + (x[1]-ay)*(x[1]-ay) + (x[2]-az)*(x[2]-az)))) - k*Z*exp(-2.0*k*sqrt((x[0]-ax)*(x[0]-ax) + (x[1]-ay)*(x[1]-ay) + (x[2]-az)*(x[2]-az))):(-3*k*Z)',
                                         k=k,ax=cartcoords[i][0],ay=cartcoords[i][1],az=cartcoords[i][2],Z=Z,vext=self.vext)
                    potexpr.vector()[:] += cuspexp.vector()[:]
                    #gui.plot1d(self.fel.unitcell,potexpr,"pot")
                    #gui.pause()
                chronicle.stop()                                
            else:   
                if self.vext == None:
                    self.vext = self.fel.new_scalar_field()
                    rhs.vector()[:] = 0.0
                    fe.solve_poisson(self.fel, self.vext, rhs, self.cartcoords, self.deltastrengths)
                potexpr = self.vext
                #gui.plot1d(self.fel.unitcell,[potexpr], "POT")
                #gui.pause()


            (vhk, nk) = split(u_k)

            chronicle.start("Setup functionals")
            densobj = DensityWeakForm(nk,pr)
            funcpots = 0            
            for f in self.functionals:
                if isinstance(f,tuple):
                    funcpots += Constant(f[0])*f[1].potential_weakform(densobj)
                else:
                    funcpots += f.potential_weakform(densobj)
            chronicle.stop()

            chronicle.start("Solve for hartree potential")            
            uniform_charge = self.N/self.fel.vol*4.0*pi                                    
            # First coupled equation: nabla^2 v_h = -4 pi n(r)
            F = -inner(grad(vhk), grad(vr))*dx + (4.0*pi*nk - Constant(uniform_charge))*vr*dx 
            # Second coupled equation: Ts[n] + Exc[n] + Vext(r) - mu = 0
            F = F + funcpots*dx + potexpr*pr*dx + vhk*pr*dx - Constant(fake_mu)*pr*dx

            # Newtons method            
            chronicle.start("Calculate Jacobian")
            J = derivative(F, u_k, du_trial)
            chronicle.stop()

            chronicle.start("Assemble system")
            A, b = assemble_system(J, -F, bcs_du)
            chronicle.stop()
            
            chronicle.start("Run solver")
            solve(A, du.vector(), b)
            chronicle.stop()
            
            chronicle.stop()

            # Conserve memory by reusing vectors for vh and n to keep dvh and dn
            dvh = vh
            vh = None
            dn = n
            n = None
            assign(dvh, du.sub(0))
            assign(dn, du.sub(1))

            #(dvh, dn) = split(du)
                        
            chronicle.start("Calculating error")
            avg = sum(dvh.vector().get_local())/len(dvh.vector().get_local())
            eps = numpy.linalg.norm(du.vector().get_local()-avg, ord=numpy.Inf)
            #eps = numpy.linalg.norm(du.vector().array(), ord=numpy.Inf)
            #eps = numpy.linalg.norm(n.vector().array()-n1.vector().array(), ord=numpy.Inf)
            chronicle.message('Iteration for self-consistency:',self.iters,'norm:', eps)

            if math.isnan(eps):
                if self.params.gui:
                    gui.plot1d(self.fel.unitcell,[dn,dvh], "Program stop: Residual error is NaN! (dn,dvh)")
                    gui.pause()
                raise Exception("Residual error is NaN")
            
            chronicle.stop()

            omega = self.params.basestep
            if self.params.neg_density_fix == 'maxstep':
                chronicle.start("Correct for negative electron density")
                dnvec = dn.vector()
                if (dnvec<0.0).sum() > 0:
                    maxomega = -min(numpy.divide(nlast.vector()[dnvec<0],dn.vector()[dnvec<0]))
                    if maxomega > 0 and maxomega < omega:
                        omega = maxomega/2.0
                chronicle.stop()    
            
            chronicle.start("Taking density and hartree potential step")
            assign(nlast, u_k.sub(1))
            u_k.vector()[:] = u_k.vector()[:] + omega*du.vector()[:]

            # Conserve memory by reusing vectors for n, vh also as dn, dvh
            vh = dvh 
            dvh = None
            n = dn 
            dn = None                        
            assign(vh, u_k.sub(0))
            assign(n, u_k.sub(1))

            # This affects the functional energy, so reset the functional energy cache
            self.e_functional = None
            
            chronicle.stop()    

            if self.params.neg_density_fix == 'adhoc':
                chronicle.start("Correct for negative electron density")
                nvec = n.vector()
                minval = nvec.min()
                if minval <= 0.0:
                    nvec[:]=nvec[:]-minval+0.1
                    intn = self.fel.integrate_scalar_field(n)
                    chronicle.message("Number of electrons before correction:",intn)
                    nvec[:] = nvec[:]*self.N/intn    
                    intn = self.fel.integrate_scalar_field(n)            
                    assign(u_k.sub(1),n)              
                    chronicle.message("Number of electrons after correction:",intn)
                chronicle.stop()    

            chronicle.start("Calculate hartree potential alignment correction")
            self.vh_align = -self.fel.integrate_scalar_field(vh) / self.fel.vol
            self.mu = fake_mu - self.vh_align
            chronicle.stop()    

            if self.params.gui:
                chronicle.start("Calculate integrated electron density")
                intn = self.fel.integrate_scalar_field(n)
                chronicle.stop()    
    
                #chronicle.start("calculate electron density gradient and laplacian")
                #gr2, la = self.fel.grad_lapl(n)
                #chronicle.stop()
    
                self.calculate_energies()
                convfact = self.convfact_energy
                self.data = ['i:%d'%(self.iters,),'err: %.4e'%(eps,),'','N:%.4f'%(intn),'mu:%.4f'%(self.mu,),
                        '$n(0)$:%.4f'%(n([0.0,0.0,0.0]),),
                        '$n(r_1)/Z_1^3$:%.4f'%(n(cartcoords[0])/self.atoms[0]**3,),
                        '$-E/Z_1^{7/3}$:%.4f'%(-(self.e_ionelec+self.e_elecelec+self.e_functional)/self.atoms[0]**(7.0/3.0),),
                        '$-E/E_{TF}$:%.4f'%((self.e_ionelec+self.e_elecelec+self.e_functional)/(-0.768745*self.atoms[0]**(7.0/3.0)),)]
                self.data += ['','$E_{ii}$: %.4f'%(self.e_ionion*convfact,),'$E_{ee}$: %.4f'%(self.e_elecelec*convfact,),
                         '$E_{ie}$: %.4f'%(self.e_ionelec*convfact,),'$E_{f}$: %.4f'%(self.e_functional*convfact,),
                         '','$E_{tot}$: %.4f'%(self.e_tot*convfact,)]
 
                chronicle.start("Plotting")
                gui.plot1d(self.fel.unitcell,[n,nlast], "Electron Density (new: blue, old: green)", self.data)
                chronicle.stop()
            if self.params.dynrefine != False:
                chronicle.start("Dynamically refine grid")
                energy_density = self.calculate_functional_energy_density()
                self.e_functional = assemble(energy_density*dx)
                self.fel.refine_mesh_integratedvalue(vh,self.params.dynrefine)
                #self.fel.refine_mesh_integratedvalue(energy_density,self.params.dynrefine)
                #egrad, elapl = self.fel.grad_lapl(energy_density)
                #self.fel.refine_mesh(energy_density, egrad, elapl,self.params.dynrefine)
                #egrad, elapl = self.fel.grad_lapl(vh)
                #self.fel.refine_mesh(vh, egrad, elapl,self.params.dynrefine)
                # Reset things that are affected by updating the mesh
                self.vext = None
                W = self.fel.W
                V = self.fel.V
                rhs = self.fel.new_scalar_field()
                vh = self.fel.new_scalar_field()
                n = self.fel.new_scalar_field()
                nlast = self.fel.new_scalar_field()
                mixed_test_functions = TestFunction(W)
                (vr,pr) = split(mixed_test_functions)
                du_trial = TrialFunction(W)        
                du = Function(W)
                chronicle.stop()
                chronicle.start("Transfer solution to refined grid")
                u_k = self.fel.transfer_mixed_scalar_field(u_k)                
                assign(vh, u_k.sub(0))
                self.vh = vh
                assign(n, u_k.sub(1))
                self.n = n
                assign(nlast, n)
                chronicle.stop()
                                                  


                                           
        chronicle.message("Iteration finished, eps:",eps)
           
        self.n = n
        vh.vector()[:] += self.vh_align
        self.vh_align = 0.0
        self.vh = vh

        chronicle.stop("Solve the system using newton solver")

        
    def show_density(self):
        import pydefuse.gui
        gui.plot1d(self.fel.unitcell,[self.n],"Final density",self.data)
        gui.pause()

    def show_vh(self):
        import pydefuse.gui
        calc_vh = self.fel.new_scalar_field()
        rhs = self.fel.new_scalar_field()
        rhs.assign(self.n)
        calc_vh.vector()[:]=0.0
        fe.solve_poisson(self.fel, calc_vh, rhs, preserve_rhs=False)
        gui.plot1d(self.fel.unitcell,[self.vh,calc_vh],"Final hartree potential (solution vs. nabla^2 n)",self.data)
        gui.pause()

    def calculate_functional_energy_density(self):    
        v = TestFunction(self.fel.V)
        #self.n = interpolate(Constant(self.N/self.fel.vol), self.fel.V)
        densobj = DensityWeakForm(self.n,v)
        funcpots = 0
        for f in self.functionals:
            if isinstance(f,tuple):
                funcpots += Constant(f[0])*f[1].energy_weakform(densobj)
            else:
                funcpots += f.energy_weakform(densobj)

        energydensity = self.fel.new_scalar_field()
        u = TrialFunction(self.fel.V)
        v = TestFunction(self.fel.V)        
        a = u*v*dx
        L = funcpots*dx
        solve(a == L, energydensity ,[])
        return energydensity


    def calculate_energies(self):    
        chronicle.start("Calculate energies")

        field = self.fel.new_scalar_field()
        tmp = self.fel.new_scalar_field()
        rhs = self.fel.new_scalar_field()
            
        if self.e_ionion == None:
            chronicle.start("Calculate ion-ion energy")
    
            if self.pbc == [False, False, False]:
                self.e_ionion = 0.0
                for i in range(len(self.atoms)):
                    ax, ay, az = self.cartcoords[i]
                    for j in range(i+1,len(self.atoms)):
                        bx, by, bz = self.cartcoords[i]
                        self.e_ionion = self.atoms[i]*self.atoms[j]/sqrt((bx-ax)*(bx-ax) + (by-ay)*(by-ay) + (bz-az)*(bz-az))
                #self.e_ionion = self.e_ionion/(8.0*pi)
            else:                

                if self.params.ionion == 'gaussians':
                    beta = 4.0
                    allexprs = 0
                    chronicle.start("Calculate ion-ion energy with gaussians")                        
                    for i in range(len(self.atoms)):
                        #allexprs += self.fel.expression("(Z*beta*beta*beta)/sqrt(pi*pi*pi)*exp(-beta*beta*((x[0]-ax)*(x[0]-ax) + (x[1]-ay)*(x[1]-ay) + (x[2]-az)*(x[2]-az))) - Z/vol",
                        #                                 Z=self.atoms[i],beta=beta,ax=self.cartcoords[i][0],ay=self.cartcoords[i][1],az=self.cartcoords[i][2],vol=self.fel.vol)
                        allexprs += self.fel.expression("(Z*beta*beta*beta)/sqrt(pi*pi*pi)*exp(-beta*beta*((x[0]-ax)*(x[0]-ax) + (x[1]-ay)*(x[1]-ay) + (x[2]-az)*(x[2]-az)))",
                                                         Z=self.atoms[i],beta=beta,ax=self.cartcoords[i][0],ay=self.cartcoords[i][1],az=self.cartcoords[i][2],vol=self.fel.vol)
                        # Doesn't nessecarily need to be gaussians
                        #allexprs += self.fel.expression("(Z*beta*beta*beta)/5.438414862332862467896027733801508004492751524320098201150*exp(-beta*beta*beta*beta*((x[0]-ax)*(x[0]-ax)*(x[0]-ax)*(x[0]-ax) + (x[1]-ay)*(x[1]-ay)*(x[1]-ay)*(x[1]-ay) + (x[2]-az)*(x[2]-az)*(x[2]-az)*(x[2]-az)))",
                        #                                 Z=self.atoms[i],beta=beta,ax=self.cartcoords[i][0],ay=self.cartcoords[i][1],az=self.cartcoords[i][2])
                        #beta = 1.5
                        #allexprs += self.fel.expression("Z*beta*beta*beta*max(peak-beta*sqrt((x[0]-ax)*(x[0]-ax) + (x[1]-ay)*(x[1]-ay) + (x[2]-az)*(x[2]-az)),0.0)",
                        #                                 peak=1.0,Z=self.atoms[i],beta=beta,ax=self.cartcoords[i][0],ay=self.cartcoords[i][1],az=self.cartcoords[i][2])
    
                    tmp = self.fel.scalar_field_from_expression(allexprs)
        
                    rhs.assign(tmp)
                    field.vector()[:] = 0.0
                    fe.solve_poisson(self.fel, field, rhs, boundaryexpr=self.boundaryexpr)
                    self.e_ionion = 0.5*self.fel.integrate_scalar_field(tmp*field)
                    chronicle.stop("Reciprocal part")                        

                    ## Correct for self energy
                    for i in range(len(self.atoms)):
                        self.e_ionion -= beta/sqrt(pi)*self.atoms[i]**2                

                elif self.params.ionion == 'ewald':                            
                    chronicle.start("Calculate ion-ion energy via gaussians")        
                    # Same idea as Ewald sum, only, the reciprocal sum can be done as a poissons solution instead.
                    # Also, with beta quite large, the short range sum is actually very small.
                    #beta = 4.0
                    #beta = 3.0
                    #beta = 3.5
                    beta = 4.0
                    allexprs = 0
                    chronicle.start("Reciprocal part")                        
                    for i in range(len(self.atoms)):
                        #allexprs += self.fel.expression("(Z*beta*beta*beta)/sqrt(pi*pi*pi)*exp(-beta*beta*((x[0]-ax)*(x[0]-ax) + (x[1]-ay)*(x[1]-ay) + (x[2]-az)*(x[2]-az))) - Z/vol",
                        #                                 Z=self.atoms[i],beta=beta,ax=self.cartcoords[i][0],ay=self.cartcoords[i][1],az=self.cartcoords[i][2],vol=self.fel.vol)
                        allexprs += self.fel.expression("(Z*beta*beta*beta)/sqrt(pi*pi*pi)*exp(-beta*beta*((x[0]-ax)*(x[0]-ax) + (x[1]-ay)*(x[1]-ay) + (x[2]-az)*(x[2]-az)))",
                                                         Z=self.atoms[i],beta=beta,ax=self.cartcoords[i][0],ay=self.cartcoords[i][1],az=self.cartcoords[i][2],vol=self.fel.vol)
                        # Doesn't nessecarily need to be gaussians
                        #allexprs += self.fel.expression("(Z*beta*beta*beta)/5.438414862332862467896027733801508004492751524320098201150*exp(-beta*beta*beta*beta*((x[0]-ax)*(x[0]-ax)*(x[0]-ax)*(x[0]-ax) + (x[1]-ay)*(x[1]-ay)*(x[1]-ay)*(x[1]-ay) + (x[2]-az)*(x[2]-az)*(x[2]-az)*(x[2]-az)))",
                        #                                 Z=self.atoms[i],beta=beta,ax=self.cartcoords[i][0],ay=self.cartcoords[i][1],az=self.cartcoords[i][2])
                        #beta = 1.5
                        #allexprs += self.fel.expression("Z*beta*beta*beta*max(peak-beta*sqrt((x[0]-ax)*(x[0]-ax) + (x[1]-ay)*(x[1]-ay) + (x[2]-az)*(x[2]-az)),0.0)",
                        #                                 peak=1.0,Z=self.atoms[i],beta=beta,ax=self.cartcoords[i][0],ay=self.cartcoords[i][1],az=self.cartcoords[i][2])
    
                    tmp = self.fel.scalar_field_from_expression(allexprs)
    
                    #import gui
                    #gui.plot1d(self.fel.unitcell,[tmp], "Gurk")
                    #gui.pause()   
    
                    rhs.assign(tmp)
                    field.vector()[:] = 0.0
                    fe.solve_poisson(self.fel, field, rhs, boundaryexpr=self.boundaryexpr)
                    self.e_ionion = 0.5*self.fel.integrate_scalar_field(tmp*field)
                    chronicle.stop("Reciprocal part")                        
    
                    def direct_ewald_uc_sum():
                        val = 0.0
                        for i in range(len(self.atoms)):
                            ax, ay, az = self.cartcoords[i]
                            for j in range(i+1,len(self.atoms)):
                                bx, by, bz = self.cartcoords[i]
                                l = sqrt((bx-ax)**2 + (by-ay)**2 + (bz-az)**2)
                                val += self.atoms[i]*self.atoms[j]*math.erfc(beta*l)/l
                        return val
    
                    def direct_ewald_offset_sum(off):
                        val = 0.0
                        othercoords = self.cartcoords + numpy.dot(off,self.unitcell)
                        for i in range(len(self.atoms)):
                            ax, ay, az = self.cartcoords[i]
                            for j in range(len(self.atoms)):
                                bx, by, bz = othercoords[i]
                                l = sqrt((bx-ax)**2 + (by-ay)**2 + (bz-az)**2)
                                val += 0.5*self.atoms[i]*self.atoms[j]*math.erfc(beta*l)/l
                        return val
    
                    chronicle.start("Direct part")                        
                    ionion_direct = direct_ewald_uc_sum()                
                    chronicle.message("Direct ion-ion energy for shell 0:",ionion_direct)
                    shell = 1
                    while True:
                        ionion_direct_shell = 0.0
                        for na in range(-shell,shell+1):
                            for nb in range(-shell,shell+1):
                                ionion_direct_shell += direct_ewald_offset_sum(numpy.array([na,nb,-shell]))
                                ionion_direct_shell += direct_ewald_offset_sum(numpy.array([na,nb,shell]))
                        for na in range(-shell+1,shell):
                            for nb in range(-shell,shell+1):
                                ionion_direct_shell += direct_ewald_offset_sum(numpy.array([na,-shell,nb]))
                                ionion_direct_shell += direct_ewald_offset_sum(numpy.array([na,shell,nb]))
                        for na in range(-shell+1,shell):
                            for nb in range(-shell+1,shell):
                                ionion_direct_shell += direct_ewald_offset_sum(numpy.array([-shell,na,nb]))
                                ionion_direct_shell += direct_ewald_offset_sum(numpy.array([shell,na,nb]))
                        chronicle.message("Direct ion-ion energy for shell:",shell,ionion_direct_shell)
                        shell += 1
                        ionion_direct += ionion_direct_shell
                        if ionion_direct_shell < 1e-12:
                            break
                    chronicle.message("Direct ion-ion energy:",ionion_direct)                    
                    self.e_ionion += ionion_direct                        
                    chronicle.stop("Direct part")                        

                    chronicle.stop("Calculate ion-ion energy via gaussians")

                    ## Correct for self energy
                    for i in range(len(self.atoms)):
                        self.e_ionion -= beta/sqrt(pi)*self.atoms[i]**2                


                elif self.params.ionion == 'external':
                    chronicle.start("Calculate Ewald sum")        
                    from pydefuse.ewald import calculate_ewald_sum
                    self.e_ionion = calculate_ewald_sum(self.unitcell, self.atoms, self.coords)
                    chronicle.stop()
                
            
                
                
                #import gui
                #gui.plot1d(self.fel.unitcell,[field2,self.vext], "Gurk")
                #gui.pause()
                #pylab.ioff()
                #pylab.show()
                #pylab.ion()                       
                #rhs.assign(self.n)
                #field.vector()[:] = 0.0
                #rhs.vector()[:] = 0.0
                #fe.solve_poisson(self.fel, field, rhs, deltacoords=self.cartcoords, deltastrengths=self.deltastrengths, boundaryexpr=self.boundaryexpr)
                #self.e_ionion = 0.5*self.fel.integrate_scalar_field(self.n*field)
                #import gui
                #gui.plot1d(self.fel.unitcell,[field], "vh + vext")
                #gui.pause()
                #pylab.ioff()
                #pylab.show()
                #pylab.ion()               


                # Screened atom idea. Does not seem to work?
                #field.vector()[:] = 0.0 
                #rhs.vector()[:] = 0.0 
                #k = 1.0
                #hk2 = self.fel.constant(-4.0*k**2) 
                #fe.solve_poisson_minus_helmholtz(self.fel,field,rhs,hk2,self.cartcoords,self.deltastrengths,tmp=tmp)   

                #import gui
                #gui.plot1d(self.fel.unitcell,[field,self.vext],"poisson - helmholtz")
                #gui.pause()
                #field.vector()[:] = field.vector() + 2.0*k*sum(self.atoms)
                #self.e_ionion2 = 0.0    
                #for coord in self.cartcoords:
                #    self.e_ionion2 += field(coord)
                #print "COMPARE E_IONION",self.e_ionion, self.e_ionion2

                # http://arxiv.org/pdf/1409.3191.pdf
                # Self energy of individual atoms
                #for atom in self.atoms:
                #    C = (atom*beta*beta*beta)/sqrt(pi*pi*pi)
                #    #self.e_ionion -= (sqrt(pi**5/(2.0*beta**10))*C**2)
            chronicle.stop()


        chronicle.start("Calculate classical elec-elec energy (hartree term)")        
        field.vector()[:] = self.vh_align + self.vh.vector()[:]
        self.e_elecelec = 0.5*self.fel.integrate_scalar_field(field*self.n)
        chronicle.stop()
        
        chronicle.start("Calculate ion-electron energy")
        self.e_ionelec = 0.0    
        for i in range(len(self.atoms)):
            self.e_ionelec -= self.atoms[i]*field(self.cartcoords[i])
        chronicle.stop()
        # Alternative implementation, gives same thing, but more expensive
        #field.vector()[:]=0.0
        #rhs.assign(field)
        #fe.solve_poisson(self.fel, field, rhs, deltacoords=self.cartcoords, deltastrengths=self.deltastrengths)
        #self.e_ionelec = self.fel.integrate_scalar_field(field*self.n)# - self.e_elecelec
                    
#         for i in range(len(self.functionals)):
#             v = TestFunction(self.fel.V)
#             densobj = DensityWeakForm(self.n,v)
# 
#             if isinstance(self.functionals[i],tuple):
#                 funcpots = Constant(self.functionals[i][0])*self.functionals[i][1].energy_weakform(densobj)
#                 name = self.functionals[i][1].__class__.__name__
#             else:
#                 funcpots = self.functionals[i].energy_weakform(densobj)
#                 name = self.functionals[i].__class__.__name__
#     
#             energydensity = self.fel.new_scalar_field()
#             u = TrialFunction(self.fel.V)
#             v = TestFunction(self.fel.V)        
#             a = u*v*dx
#             L = funcpots*dx
#             solve(a == L, energydensity ,[])
#             
#             val = assemble(energydensity*dx)
#             print "Functional:",name,"val:",val
                    
        chronicle.start("Calculate functional energy")        
        if self.e_functional == None:
            energydensity = self.calculate_functional_energy_density()
            self.e_functional = assemble(energydensity*dx)
        chronicle.stop()

        self.e_tot = self.e_ionion + self.e_ionelec + self.e_elecelec + self.e_functional
        chronicle.stop("Calculate energies")

    def print_energies(self):
        convfact = self.convfact_energy            
        print ("==== Resulting energies ("+self.convfact_energy_name+"): ================")
        print ("Ion-ion:        % 10.4f"%(self.e_ionion*convfact,))
        print ("Ion-elec:       % 10.4f"%(self.e_ionelec*convfact,))
        print ("Elec-elec (Eh): % 10.4f"%(self.e_elecelec*convfact,))
        print ("Functional:     % 10.4f"%(self.e_functional*convfact,))
        print ("==============================================")
        print ("Total energy:   % 10.4f"%(self.e_tot*convfact,))
        print ("Total w/o i-i:  % 10.4f"%((self.e_ionelec+self.e_elecelec+self.e_functional)*convfact,))
        print ("==============================================")
        print()
        
    def exit(self):
        if self.params.gui:
            import pydefuse.gui
            gui.gui_exit()
        
        
        
