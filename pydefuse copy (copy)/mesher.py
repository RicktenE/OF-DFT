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

import subprocess, tempfile, os

import numpy
from dolfin import *
# from scipy.spatial import Delaunay
# I have no idea why scipy.spatial.Delaunay is not working
# One get meshes that Dolfin just give NaN solutions for
# I should probably ask someone about that.

# Adapted from
# http://codereview.stackexchange.com/questions/41024/faster-computation-of-barycentric-coordinates-for-many-points
# cc by-sa 3.0 with attribution required
# Gareth Rees : http://codereview.stackexchange.com/users/11728/gareth-rees
def tetrahedralization_tetgen(points, facets = None, holes = None, extra_surf_points=False):
    #print "POINTS",points
    #print "FACETS",facets
    #print "HOLES",holes

    if extra_surf_points:
        switches='-z'
    else:
        switches='-zpqY'

    if facets == None and holes == None:
        node_f = tempfile.NamedTemporaryFile(suffix=".node", delete=False);
        style='node'
    else:
        if len(facets) == 1:
            node_f = tempfile.NamedTemporaryFile(suffix=".smesh", delete=False);
            style='smesh'
        else:
            node_f = tempfile.NamedTemporaryFile(suffix=".poly", delete=False);
            style='poly'
    node_f.write("%i 3 0 0\n" % len(points))
    for i, point in enumerate(points):
        node_f.write("%i %f %f %f\n" % (i, point[0], point[1], point[2]))
    if style != 'node':
        if style == 'poly':
            node_f.write("\n%i 0\n" % (len(facets),))            
        else:
            facets = [reduce(lambda x,y:x+y,facets)]
        for facet in facets:
            node_f.write("%i\n" % (len(facet,)))
            for poly in facet:
                node_f.write("%i " % (len(poly),))
                node_f.write(" ".join(["%d" %(x,) for x in poly])+"\n")
        node_f.write("\n%i\n" % len(holes))
        for i, hole in enumerate(holes):
            node_f.write("%i %f %f %f\n" % (i, hole[0], hole[1], hole[2]))
        node_f.write("\n0\n")
    node_f.close()

    #print "STORE IN:",node_f.name,"SWITCHES",switches
    
    #subprocess.call(["tetgen", node_f.name], stdout=open(os.devnull, 'wb'))
    subprocess.call(["tetgen", switches, node_f.name])        
    ele_f_name = node_f.name[:-(len(style)+1)] + ".1.ele"
    face_f_name = node_f.name[:-(len(style)+1)] + ".1.face"
    node_f_name = node_f.name[:-(len(style)+1)] + ".1.node"

    ele_f_lines = [line.split() for line in open(ele_f_name)][1:-1]
    face_f_lines = [line.split() for line in open(face_f_name)][1:-1]
    node_f_points = [line.split() for line in open(node_f_name)][1:-1]

    tets = [map(int, line[1:]) for line in ele_f_lines]
    facets = [[map(int, line[1:]) for line in face_f_lines]]
    verts = [map(float, line[1:4]) for line in node_f_points]

    return verts, tets, facets 


def tetrahedralization_meshpy(points, facets = None, holes = None, extra_surf_points=False):
    import meshpy.tet

    if extra_surf_points:
        meshpy.tet.Options(switches='z')
    else:
        meshpy.tet.Options(switches='zpYq')
      
    mesh_info = meshpy.tet.MeshInfo()
    mesh_info.set_points(points)
    # Merge facets?
    if facets != None and len(facets)>0:
        facets = reduce(lambda x,y:x+y,facets)
        mesh_info.set_facets(facets)
    if holes != None and len(holes)>0:
        mesh_info.set_holes(holes)
    mesh = meshpy.tet.build(mesh_info)
        
    return list(mesh.points), list(mesh.elements), list(mesh.facets)


def build_mesh(verts,cells):
    dim = 3
    msh = Mesh()
    editor = MeshEditor()
    editor.open(msh,dim,dim)
    editor.init_vertices(len(verts))
    i_vert = 0
    for vert in verts:
        editor.add_vertex(i_vert, *vert)
        i_vert +=1
    i_cell = 0
    editor.init_cells(len(cells))
    for cell in cells:
        editor.add_cell(i_cell, *cell)
        i_cell += 1
    editor.close()    
    return msh


def unit_rect_verticies(nx,ny,nz,filled=True):
    xpos = numpy.linspace(0.0, 1.0, nx+1)
    ypos = numpy.linspace(0.0, 1.0, ny+1)    
    zpos = numpy.linspace(0.0, 1.0, nz+1)    
    verts = []
    facets = [[],[],[],[],[],[]]

    idx = 0        
    lookup = {}
    for ix in range(nx+1):        
        for iy in range(ny+1):
            for iz in range(nz+1):
                if ix == 0 or ix == ny or \
                   iy == 0 or iy == ny or \
                   iz == 0 or iz == nz:
                    lookup[(ix,iy,iz)] = idx
                    verts += [(xpos[ix ],ypos[iy],zpos[iz])]
                else:
                    if filled:
                        verts += [(xpos[ix ],ypos[iy],zpos[iz])]
                idx += 1
    
    for ix in range(nx):        
        for iy in range(ny):
            facets[0] += [(lookup[(ix,iy,0)],lookup[(ix+1,iy,0)],lookup[(ix+1,iy+1,0)],lookup[(ix,iy+1,0)])]
            facets[3] += [(lookup[(ix,iy,nz)],lookup[(ix+1,iy,nz)],lookup[(ix+1,iy+1,nz)],lookup[(ix,iy+1,nz)])]
            
    for ix in range(nx):        
        for iz in range(nz):
            facets[1] += [(lookup[(ix,0,iz)],lookup[(ix+1,0,iz)],lookup[(ix+1,0,iz+1)],lookup[(ix,0,iz+1)])]
            facets[4] += [(lookup[(ix,ny,iz)],lookup[(ix+1,ny,iz)],lookup[(ix+1,ny,iz+1)],lookup[(ix,ny,iz+1)])]
            
    for iy in range(nx):        
        for iz in range(nz):
            facets[2] += [(lookup[(0,iy,iz)],lookup[(0,iy+1,iz)],lookup[(0,iy+1,iz+1)],lookup[(0,iy,iz+1)])]
            facets[5] += [(lookup[(nx,iy,iz)],lookup[(nx,iy+1,iz)],lookup[(nx,iy+1,iz+1)],lookup[(nx,iy,iz+1)])]
        
    return verts,facets


def unitcell_verticies(unitcell,nx,ny,nz,filled=True):
    verts, facets = unit_rect_verticies(nx,ny,nz,filled)
    return numpy.dot(verts, unitcell), facets


def verts_remove_spheres(verts,facets,center_coords,radii):
    def near_sphere(v):
        for s,r in zip(center_coords,radii):
            if numpy.linalg.norm(v-s) <= r:
                return True
        return False
    new_verts = []
    for idx in range(len(verts)):
        if near_sphere(verts[idx]):
            #print facets
            facets = [[[x-1 if x>=idx else x for x in poly] for poly in f] for f in facets]
        else:
            new_verts += [verts[idx]]
            
    return new_verts, facets
        

def sphere_surface(center_coords,radius,refine):

    t = (1.0 + sqrt(5.0))/2.0

    vertices = numpy.array([[-1,  t,  0],[ 1,  t,  0],[-1, -t,  0],[ 1, -t,  0],
                [ 0, -1,  t],[ 0,  1,  t],[ 0, -1, -t],[ 0,  1, -t],
                [ t,  0, -1],[ t,  0,  1],[-t,  0, -1],[-t,  0,  1]])
    
    vertices = [x/sqrt(x[0]**2+x[1]**2+x[2]**2) for x in vertices]

    cells = [[0, 11, 5],[0, 5, 1],[0, 1, 7],[0, 7, 10],[0, 10, 11],
             [1, 5, 9],[5, 11, 4],[11, 10, 2],[10, 7, 6],[7, 1, 8],
             [3, 9, 4],[3, 4, 2],[3, 2, 6],[3, 6, 8],[3, 8, 9],
             [4, 9, 5],[2, 4, 11],[6, 2, 10],[8, 6, 7],[9, 8, 1]];

    seen = {}
    def get_midpoint(p1,p2,vertices):
        if (p1,p2) in seen:
            return seen[(p1,p2)]
        elif (p2,p1) in seen:
            return seen[(p2,p1)]
        nv = (vertices[p1] + vertices[p2])/2.0
        nv = nv/sqrt(nv[0]**2+nv[1]**2+nv[2]**2)
        idx = len(vertices)
        seen[(p1,p2)] = idx
        vertices += [nv]
        return idx
    
    for i in range(refine):
        new_cells = []
        for cell in cells:
            a = get_midpoint(cell[0],cell[1],vertices)
            b = get_midpoint(cell[1],cell[2],vertices)
            c = get_midpoint(cell[2],cell[0],vertices)
            new_cells += [(cell[0],a,c),(cell[1],b,a),(cell[2],c,b),(a,b,c)]
        cells = new_cells

    vertices = list(numpy.array(vertices)*radius + center_coords)

    return vertices, cells


def add_symmetric_sphere(center_coords,verts,surf_verticies_idx,surf_verticies_count,cells,surf_cells,rdivs=2):    
   
    i_vert = len(verts)

    verts += [center_coords]
    center_idx = i_vert
    i_vert += 1
            
    boundary_vertex_map = {}
    for i in range(surf_verticies_count):   
        idx = surf_verticies_idx+i
        vertex = verts[idx]
        radial_idxs = []
        for j in range(1,rdivs):
            #radial_coords = center_coords + ((1.0*j)/(rdivs+1))**2*(vertex-center_coords)
            radial_coords = center_coords + (1.0*j)/(rdivs+1)*(vertex-center_coords)
            verts += [radial_coords]            
            radial_idxs += [i_vert]
            i_vert+=1
        radial_idxs += [idx]
        boundary_vertex_map[i] = radial_idxs

    for cell_vert_idxs in surf_cells:
        bcs = [boundary_vertex_map[x] for x in cell_vert_idxs]
        # First cell:
        cells += [list([center_idx] + [x[0] for x in bcs])]
        # Rest
        for i in range(1,rdivs):
            prism = numpy.array((bcs[0][i-1],bcs[1][i-1],bcs[2][i-1],bcs[2][i],
                    bcs[0][i],bcs[1][i]))
            for split in ([0,1,2,3],[1,2,3,4],[2,3,4,5]):                        
                cells += [list(sorted(prism[split]))]                        


def jigzaw_icosahedron_mesher(unitcell, centers, radii, uc_divs=6, sphere_divs=1, sphere_rad_divs=6):
    ir_holes = centers
    verts, facets = unitcell_verticies(unitcell,uc_divs,uc_divs,uc_divs,filled=False)
    #verts, cells, ir_facets = tetrahedralization_tetgen_old(verts)
    verts, cells, ir_facets = tetrahedralization_tetgen(verts,extra_surf_points=True)
    #verts, cells, ir_facets = tetrahedralization_meshpy(verts,extra_surf_points=True)
    # Not needed if we just mesh exterior points
    #verts, ir_facets = verts_remove_spheres(verts, ir_facets, centers, radii)
    #verts, cells, ir_facets = tetrahedralization_meshpy(verts,[],[])

    vshift = len(verts)
    vert_count_atom=[]
    surf_cells_atom=[]
    idx = len(verts)
    for atomidx in range(len(centers)):
        mt_surf_verts, mt_surf_cells = sphere_surface(centers[atomidx],radii[atomidx],sphere_divs)
        verts += mt_surf_verts
        ir_facets += [[[x+idx for x in f] for f in mt_surf_cells]]
        idx += len(mt_surf_verts)

        vert_count_atom += [len(mt_surf_verts)]
        surf_cells_atom += [mt_surf_cells]
    #print "Meshing:",len(verts),"verticies in interstitial region"

    # It seems tetgen likes this better
    ir_facets = [reduce(lambda x,y:x+y,ir_facets)]

    outverts, cells, _dummy = tetrahedralization_tetgen(verts,ir_facets,ir_holes)
    #outverts, cells, _dummy = tetrahedralization_meshpy(verts,ir_facets,ir_holes)
    outverts = [[x for x in list(y)] for y in list(outverts)]
    cells = [[x for x in list(y)] for y in list(cells)]
        
    for i in range(len(verts)):
        #print verts[i], outverts[i]
        if numpy.linalg.norm(numpy.array(verts[i]) - numpy.array(outverts[i])) > 1e-6:
            #> DOLFIN_EPS:
            print ("ERR",verts[i],outverts[i])
            raise Exception("Tetrahedralization did not give the same points back")
    
    verts = outverts
    
    for atomidx in range(len(centers)):   
        add_symmetric_sphere(centers[0],verts,vshift,vert_count_atom[atomidx],cells,surf_cells_atom[atomidx],rdivs=sphere_rad_divs)
        vshift += vert_count_atom[atomidx]
    
    msh = build_mesh(verts,cells)

    #plot(msh)
    #interactive()

    return msh

