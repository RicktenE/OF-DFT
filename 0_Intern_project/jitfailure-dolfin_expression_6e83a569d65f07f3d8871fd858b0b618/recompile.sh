#!/bin/bash
# Execute this file to recompile locally
c++ -Wall -shared -fPIC -std=c++11 -O3 -fno-math-errno -fno-trapping-math -ffinite-math-only -I/usr/include/scotch -I/usr/lib/x86_64-linux-gnu/hdf5/openmpi/include -I/usr/include/suitesparse -I/usr/include/superlu -I/usr/include/hypre -I/usr/lib/petscdir/3.7.7/x86_64-linux-gnu-real/include -I/usr/lib/slepcdir/3.7.4/x86_64-linux-gnu-real/include -I/usr/lib/x86_64-linux-gnu/openmpi/include -I/usr/lib/x86_64-linux-gnu/openmpi/include/openmpi/opal/mca/event/libevent2022/libevent/include -I/usr/lib/x86_64-linux-gnu/openmpi/include/openmpi/opal/mca/event/libevent2022/libevent -I/usr/lib/x86_64-linux-gnu/openmpi/include/openmpi -I/usr/include/hdf5/openmpi -I/usr/include/eigen3 -I/home/rick/.cache/dijitso/include dolfin_expression_6e83a569d65f07f3d8871fd858b0b618.cpp -L/usr/lib/x86_64-linux-gnu/openmpi/lib -L/usr/lib/petscdir/3.7.7/x86_64-linux-gnu-real/lib -L/usr/lib/slepcdir/3.7.4/x86_64-linux-gnu-real/lib -L/usr/lib/x86_64-linux-gnu/hdf5/openmpi -L/home/rick/.cache/dijitso/lib -Wl,-rpath,/home/rick/.cache/dijitso/lib -lmpi -lmpi_cxx -lpetsc_real -lslepc_real -lm -ldl -lz -lsz -lhdf5 -lboost_timer -ldolfin -olibdijitso-dolfin_expression_6e83a569d65f07f3d8871fd858b0b618.so