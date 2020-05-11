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

import pylab, matplotlib
import numpy

pylab_remember_interactive = None

def gui_init():
    global pylab_remember_interactive
    pylab_remember_interactive = matplotlib.is_interactive()
    pylab.ion()

def gui_exit():
    global pylab_remember_interactive
    if pylab_remember_interactive:
        pylab.ion()
    else:
        pylab.ioff()

def plot1d(unitcell, fields,title="", data = None):

    if not isinstance(fields,list):
        fields = [fields]
        
    corner = unitcell[0] + unitcell[1] + unitcell[2]
    length = numpy.linalg.norm(corner)
    steps = 200
    #steps = int(10.0*length/2.0)*2.0
    r = range(int(steps+1))
    xdists = [float(i)/steps*length for i in r]
    xvals = [[float(i)/steps*corner[0],float(i)/steps*corner[1],float(i)/steps*corner[2]] for i in r]
    pylab.figure(1)
    pylab.clf()
    pylab.title(title)
    for i in range(len(fields)):
        yvals = [fields[i](x) for x in xvals]
        pylab.plot(xdists,yvals,'x-')

    if data != None:
        textstr = ''
        for d in data:
            textstr += str(d)+"\n"
        textstr = textstr[:-2]
        
        props = dict(facecolor='white', alpha=0.5)
        pylab.axes().text(0.05, 0.95, textstr, transform=pylab.axes().transAxes, fontsize=14,
            verticalalignment='top', bbox=props)

    #pylab.show()
    #pylab.axis([length/2.0-5.0,length/2.0+5.0,-5,5])
    pylab.draw()
    pylab.pause(0.0001)
    #pylab.waitforbuttonpress()
    #pylab.savefig('density.pdf')
    
    
def plot_radial(rs, fields,title="", data = None, xs=None):

    if not isinstance(fields,list):
        fields = [fields]

    if xs == None:
        xs = rs
        
    pylab.figure(1)
    pylab.clf()
    pylab.title(title)
    for i in range(len(fields)):
        yvals = [fields[i](x) for x in rs]
        pylab.plot(xs,yvals,'x-')

    if data != None:
        textstr = ''
        for d in data:
            textstr += str(d)+"\n"
        textstr = textstr[:-2]
        
        props = dict(facecolor='white', alpha=0.5)
        pylab.axes().text(0.05, 0.95, textstr, transform=pylab.axes().transAxes, fontsize=14,
            verticalalignment='top', bbox=props)

    #pylab.show()
    #pylab.xlim([0,100])
    pylab.draw()
    pylab.pause(0.0001)
    #pylab.waitforbuttonpress()
    #pylab.savefig('density.pdf')
    
    
    
    
def pause():
    print( "**** click on figure to continue *****")
    pylab.waitforbuttonpress()
