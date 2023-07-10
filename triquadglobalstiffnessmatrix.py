import numpy as np
import concurrent.futures
from numpy.polynomial.legendre import leggauss
def vector_stiffness_matrix_CST(v,E,x1,x2,x3,y1,y2,y3):
    matrix = np.array([[1,x1,y1],[1,x2,y2],[1,x3,y3]])
    A = (np.linalg.det(matrix) / 2)
    B = np.array([[y2-y3,y3-y1,y1-y2],
                    [x3-x2,x1-x3,x2-x1]])*(1/(2*A))
    B1=np.zeros([3,6])
    B1[0, [0, 2, 4]] = B[0]
    B1[1, [1, 3, 5]] = B[1]
    B1[2, [0, 2, 4]] = B[1]
    B1[2, [1, 3 ,5]] = B[0] 
    D=np.array([[1,v,0],[v,1,0],[0,0,(1-v)/2]])*(E/(1-v**2))
    # D=np.array([[1-v,v,0],[v,1-v,0],[0,0,(1-2*v)/2]])*(E/((1+v)*(1-2*v)))
    C = np.transpose(B1)
    s = A  * np.dot(C, np.dot(D,B1)) 
    return s
def stiffness_matrix(v,E,x1,x2,x3,x4,y1,y2,y3,y4):
    def gauss_quadrature(num_points):
        psi_values, weights_psi = leggauss(num_points)
        eta_values, weights_eta = leggauss(num_points)
        return psi_values, eta_values, weights_psi,weights_eta
    num_points=2
    psi_values, eta_values,weights_psi,weights_eta=gauss_quadrature(num_points)
    D=np.array([[1,v,0],[v,1,0],[0,0,(1-v)/2]])*(E/(1-v**2))
    K_local=np.zeros([8,8])
    for j in range(2):
        for k in range(2):
            psi=psi_values[j]
            eta=eta_values[k]
            w1=weights_psi[j]
            w2=weights_eta[k]
            B1=np.zeros([3,8])
            BN=np.array([[eta-1,1-eta,1+eta,-1-eta],
                        [psi-1,-psi-1,1+psi,1-psi]])*0.25
            a=np.array([[x1,y1],[x2,y2],[x3,y3],[x4,y4]])
            J=np.dot(BN,a)
            dN = np.dot(np.linalg.inv(J),BN)
            B1[0, [0, 2, 4,6]] = dN[0]
            B1[1, [1, 3, 5,7]] = dN[1]
            B1[2, [0, 2, 4,6]] = dN[1]
            B1[2, [1, 3 ,5,7]] = dN[0] 
            J1=np.linalg.det(J)
            K_local+=w1*w2*J1*np.dot(np.transpose(B1),np.dot(D,B1))
    return K_local
def K_compute( nodes, elements,v,E):
    if len(elements[0])==3:
        k_collection = np.zeros([len(elements), 6, 6])
        for i in range(len(elements)): 
            x1,x2,x3=nodes[elements[i]][:,0]
            y1,y2,y3=nodes[elements[i]][:,1]
            k_collection[i] =vector_stiffness_matrix_CST(v,E,x1,x2,x3,y1,y2,y3)
    elif len(elements[0])==4:
        k_collection = np.zeros([len(elements), 8, 8])
        for i in range(len(elements)): 
            x1,x2,x3,x4=nodes[elements[i]][:,0]
            y1,y2,y3,y4=nodes[elements[i]][:,1]
            k_collection[i] =stiffness_matrix(v,E,x1,x2,x3,x4,y1,y2,y3,y4)
    return k_collection
def connectivity(nodes, elements, v,E,num_processes):
    nodes = np.array(nodes)
    elements = np.array(elements)
    s = len(elements[0])
    n = len(nodes)
    k_global = np.zeros([2*n,2*n])
    arr_process = np.zeros(num_processes)
    arr_process[:] = len(elements) // (num_processes)
    arr_process[:len(elements) % (num_processes)] += 1
    arr_process = arr_process.astype("int32")
    processess = []
    slicing = [0]
    slice = 0
    for i in range(num_processes):
        slice += arr_process[i]
        slicing.append(slice)
    with concurrent.futures.ProcessPoolExecutor() as executor:
        for i in range(num_processes):
            f = executor.submit(K_compute,nodes,elements[slicing[i]:slicing[i+1],:],v,E)
            processess.append(f)
    slice = 0
    for i in range(num_processes):
        k_collection = processess[i].result()
        for j in range(len(k_collection)):
            ele_node = np.array(elements[int(slice + j)])
            c = np.zeros([2 * s])
            c[np.arange(s)*2] = 2 * ele_node
            c[np.arange(s)*2+1] = 2 * ele_node + 1
            c = c.astype("int32")
            k_global[c[:, None], c] += k_collection[j]
        slice += len(k_collection)  
    return k_global