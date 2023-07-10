from triquadglobalstiffnessmatrix import connectivity
from triquadglobalforce import boundary_forces_quad
from triquaddisplacement import disp
from triquadstressstrain import element_strain
from triquadplot import plotting
from triquadplot import plotting_mesh
from triquadplot import display_quad
import meshio
import numpy as np
import matplotlib.pyplot as plt

def quadrilateral_elements():
    mesh = meshio.read('quadrihole.msh')
    nodes = mesh.points
    quads = mesh.get_cells_type("quad")
    return nodes,quads
nodes,elements=quadrilateral_elements()
elements=np.array(elements)
nodes=nodes[:,[0,1]]
# nodes = np.array([[0,0],[1,0],[2,0],[0,1],[1,1],[2,1],[0,2],[1,2],[2,2]])
# elements=[[1,2,5,4],[2,3,6,5],[5,6,9,8],[4,5,8,7]]
# elements=np.array(elements)-1
E=1e7
t=[1e5,0]
num_processes=8
v=0.3
displacements={3:[0,0],4:[0,0],25:[0,0],26:[0,0]}
loads={3:[0,0],4:[0,0],25:[0,0],26:[0,0]}
if __name__=='__main__':
    plotting_mesh(nodes,elements)
    k_global=connectivity(nodes, elements, v,E,num_processes)
    f1=boundary_forces_quad(nodes,elements,t,traction_element=37,psi = -1, eta = 0)
    f2=boundary_forces_quad(nodes,elements,t,traction_element=51,psi = -1, eta = 0)
    f3=boundary_forces_quad(nodes,elements,t,traction_element=30,psi = -1, eta = 0)
    f4=boundary_forces_quad(nodes,elements,t,traction_element=50,psi = 0, eta = -1)
    f5=boundary_forces_quad(nodes,elements,t,traction_element=36,psi = 0, eta = -1)
    U=disp(displacements,loads,nodes,elements,v,E,num_processes,[37,51,30,50,36],forces=[f1,f2,f3,f4,f5])
    total_force=np.dot(k_global,U)
    e,s=element_strain(nodes,elements,v,E,U)
    plotting(nodes,elements,U)
    plt.show()
    _,ax1= plt.subplots()
    display_quad(nodes,elements,ax1,s[:,0,:])
    plt.show()
   
