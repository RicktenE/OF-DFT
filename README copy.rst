=====================================
*PyDeFuSE* git repository README file
=====================================

|  *PyDeFuSE*
|  Python DEnsity FUnctional Solver for finite Element method
|
|  For License information see the file LICENSE.
|  Contact: pydefuse [at] openmaterialsdb.se

-----------------
About *PyDeFuSE*
-----------------

*PyDeFuSE* is a python program that solves the orbital-free DFT equations (and maybe some day regular DFT)
using finite elements. It uses the finite element libraries FEniCS and DOLFIN. 

-------------------------------
Getting started with *PyDeFuSE*
-------------------------------

Download
========

Clone the github development repository::

  > git clone https://github.com/rartino/pydefuse

Installation
============

#. You need python with numpy

#. You need FEniCS and DOLFIN installed and working 
   (``from dolfin import *`` needs to work in python) 
     (see `<http://fenicsproject.org/download/>`_ or, in Debian/Ubuntu 
     ``apt-get install fenics python-mshr python-dolfin``)

#. For Ewald sum calculations with ionion='external' (which is the default) you need pymatgen. 
   Setting ionion='gaussians' should be good enough as long as your FE mesh is fine enough, 
   but this is much less tested. 
     (see `<http://pymatgen.org/#guided-install>`_ )

#. For mesh_type='jigzaw' you need the tetgen binary available in your path 
   (typing out 'tetgen' in your terminal needs to work). 
     (see `<http://wias-berlin.de/software/tetgen/#Download>`_ or, in Debian/Ubuntu: ``apt-get install tetgen``)

#. After cloning the git repository you do not need to do anything more to run PyDeFuSE. Just run the examples.

User's guide
============

Try out the examples in the examples/ directory.

Known issues
************

* Code only really tested with periodic boundary conditions ``pbc=[True,True,True]``

* Radial part of code was working at some point but is not updated or tested since latest changes.

* Jigzaw mesh (``mesh_type='jigzaw'``) isn't well tested, and cannot handle muffin tins crossing the cell boundary. 
  Nor is there presently a way to configure the size of muffin tins.)

* Setting the number of electrons (``N=?``) to something other than None (meaning netural) is not tested. 
  It is meant to add a constant background to periodic calculations, and solve an ionic system for non-periodic BCs.

Reporting bugs
**************

We track our bugs using the issue tracker at github. 
If you find a bug, please search to see if someone else
has reported it here:

  https://github.com/rartino/pydefuse/issues

If you cannot find it already reported, please click the 'new issue' 
button and report the bug.


Developing / contributing to *PyDeFuSE*
***************************************

Please use git and github functionality to provide changes either as patch files in reported issues, or
as pull requests.


Citing *PyDeFuSE* in scientific works
*************************************

These are presently the preferred citations for *PyDeFuSE*:

* Orbital-free Density-Functional Theory in a Finite Element Basis, J. Davidsson (Master's thesis) (2015); 

* PyDeFuSE, J. Davidsson, A. Lindmaa, and R. Armiento (to be published)

Contributors
************

Rickard Armiento, Joel Davidsson, Alexander Lindmaa.


Acknowledgements
****************

*PyDeFuSE* has kindly been funded in part by:
   * The Swedish Research Council (VR) Grant No. 621-2011-4249

   * The Linnaeus Environment at Linköping on Nanoscale Functional
     Materials (LiLi-NFM) funded by the Swedish Research Council (VR).


License and redistribution
**************************

*PyDeFuSE* uses the GNU Affero General Public
License, which is an open source license that allows redistribution
and re-use if the license requirements are met. (Note that this
license contains clauses that are not in the GNU Public License, and
source code from httk thus cannot be imported into GPL licensed
projects.)

For the full license text, see the file COPYING.

Contact
*******

Our primary point of contact is email to: pydefuse [at] openmaterialsdb.se
(where [at] is replaced by @)

