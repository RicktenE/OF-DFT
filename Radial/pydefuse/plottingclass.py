#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 11 15:20:49 2020

@author: rick
"""

class plotting_(object):
    def plotting_psi(self,u,title):
        a_0 = 1 # Hatree units
        Alpha_ = (4/a_0)*(2*self.Z/(9*pi**2)**(1/3))
        
        pylab.clf()
        rplot = self.msh.coordinates()
        x = rplot*Alpha_
        y = [u(v)**(2/3)*v*(3*math.pi**2/(2*numpy.sqrt(2)))**(2/3)  for v in rplot]
        #x = numpy.logspace(-5,2,100)

        pylab.plot(x,y,'bx-')
        pylab.title(title)
        pylab.pause(0.0001)
        pylab.xlabel("Alpha * R")
        pylab.ylabel("Psi")
    
        return     
    
    def plotting_normal(self,u,title):
        pylab.clf()
        rplot = self.msh.coordinates()
        x = rplot
        #x = numpy.logspace(-5,2,100)
        y = [u(v) for v in rplot]
    
        pylab.plot(x,y,'bx-')
        pylab.title(title)
        pylab.pause(0.0001)
        pylab.xlabel("r")
        pylab.ylabel("n[r]")
    
        return 
    
    def plotting_sqrt(self, u ,title):
        pylab.clf()
        rplot = self.msh.coordinates()
        x = np.sqrt(rplot)
        #x = numpy.logspace(-5,2,100)
        y = [v*numpy.sqrt(u(v)) for v in rplot] 
       
        pylab.plot(x,y,'bx-')
        pylab.title(title)
        pylab.pause(0.0001)
        pylab.xlabel("SQRT(R)")
        pylab.ylabel("R * SQRT(density")
        
        return