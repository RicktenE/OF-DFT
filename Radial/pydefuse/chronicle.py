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


from __future__ import print_function
import time

class Chronicle(object):

    def __init__(self,loglevel=1):
        self.level=0
        self.loglevel=loglevel
        self.timertotals={}
        self.timers={}
        self.timersrunning=[]
        self.parent=''

    def start(self,name,silent=False):
        if not silent and self.level < self.loglevel:
            print("="*(8-2*self.level)+" "+name)
        if (self.parent,self.level) not in self.timers:
            self.timers[(self.parent,self.level)]={}
            self.timertotals[(self.parent,self.level)]={}
        self.timers[(self.parent,self.level)][name]=time.clock()
        self.timersrunning+=[name]
        self.level += 1
        self.parent = name

    def message(self,*msgs):
        if self.level < self.loglevel:
            print(*msgs)
    
    def stop(self,checkname=None,silent=False):
        self.level -= 1
        name = self.timersrunning.pop()
        if checkname is not None and checkname != name:
            raise Exception("Chronicle start-stop mismatch. Should be",checkname," but was:",name)
        if len(self.timersrunning) > 0:
            self.parent = self.timersrunning[-1]
        else:
            self.parent = ''
        delta = time.clock() - self.timers[(self.parent,self.level)][name]
        if name in self.timertotals[(self.parent,self.level)]:
            self.timertotals[(self.parent,self.level)][name] += delta
        else:
            self.timertotals[(self.parent,self.level)][name] = delta
        if not silent and self.level < self.loglevel:
            print("="*(8-2*self.level),name,"finished","("+str(delta),"sec)")

    def printlevel(self,level=0,parent=''):
        for timer, tot in sorted(self.timertotals[(parent,level)].items(),key=lambda x:x[1],reverse=True):
            print(" "+(" "*(2*level)),("%.4f"%(tot)),":",timer)
            if (timer,level+1) in self.timertotals and level < self.loglevel:
                self.printlevel(level+1,timer) 

    def printall(self):
        print("==== Timings (sec):")
        self.printlevel()
        print("====")

chronicle = Chronicle()
