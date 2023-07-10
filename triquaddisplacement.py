import numpy as np
from triquadglobalstiffnessmatrix import connectivity
from triquadglobalforce import global_force
def remove_row_and_column(displacements, nodes,elements,v, E,num_processes):
    matrix = connectivity(nodes,elements,v,E,num_processes)
    rows_to_remove = []
    for node, displacement in displacements.items():
        if displacement[0] is not None:
            i1 = node * 2 
            rows_to_remove.append(i1)
        if displacement[1] is not None:
            i2 = node * 2 + 1
            rows_to_remove.append(i2)
    new_matrix = np.delete(matrix, rows_to_remove, axis=0)  # remove rows
    new_matrix = np.delete(new_matrix, rows_to_remove, axis=1)  # remove columns
    return new_matrix
def remove_row(loads,nodes,elements,traction_element_list,forces):

    f=global_force(nodes,elements,traction_element_list,forces)
    for node, load in loads.items():
        idx = 2 * node
        f[idx] = load[0]
        f[idx+1] = load[1]
    rows_to_remove = []
    for node, force in loads.items():
        if force[0] is not None:
            i1 = node * 2 
            rows_to_remove.append(i1)
        if force[1] is not None:
            i2 = node * 2 +1
            rows_to_remove.append(i2)
    new_matrix = np.delete(f, rows_to_remove, axis=0) 
        
         # remove rows
        
    return new_matrix
def disp(displacements,loads,nodes,elements,v,E,num_processes,traction_element_list,forces):
    fr=remove_row(loads,nodes,elements,traction_element_list,forces)
    kg=remove_row_and_column(displacements, nodes,elements,v, E,num_processes)
    U=np.full([2*len(nodes),1],np.nan)
    for node, displacement in displacements.items():
        if displacement[0] is not None:
            i1=2*node
            U[i1]=displacement[0]
        if displacement[1] is not None:
            i2=2*node+1
            U[i2]=displacement[1]
    free=np.argwhere(np.isnan(U))[:,0]
    U[[free],:] = np.linalg.solve(kg,fr)
    return U