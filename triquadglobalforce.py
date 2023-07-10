import numpy as np
from numpy.polynomial.legendre import leggauss
def gauss_quadrature(num_points):
    psi_values, weights_psi = leggauss(num_points)
    # eta_values, weights_eta = leggauss(num_points)
    return psi_values, weights_psi
def shape_f(nodes,elements,traction_element, x, y):
    x1, x2, x3 = nodes[elements[traction_element]][:, 0]
    y1, y2, y3 = nodes[elements[traction_element]][:, 1]
    matrix = np.array([[1, x1, y1],
                        [1, x2, y2],
                        [1, x3, y3]])
    A = (np.linalg.det(matrix) / 2)
    N = np.array([[x2 * y3 - x3 * y2 + (y2 - y3) * x + (x3 - x2) * y],
                    [x3 * y1 - x1 * y3 + (y3 - y1) * x + (x1 - x3) * y],
                    [x1 * y2 - x2 * y1 + (y1 - y2) * x + (x2 - x1) * y]]) * (1 / (2 * A))
    return N
def traction_force_triangle(nodes,elements,traction_element,t,limits,y = None,x=None):
    if y is not None:
        def integral(x):
            return shape_f(nodes,elements,traction_element,x,y(x))
    else:
        def integral(y):
            return shape_f(nodes,elements,traction_element,x(y),y)
    ans = 0
    psi,w = gauss_quadrature(2)
    a = limits[0]
    b = limits[1]
    j=(b-a)/2
    for i in range(2):
        ans+= j*w[i]*integral(((a+b)/2)+((b-a)/2)*psi[i])
    fx=t[0]*ans
    fy=t[1]*ans
    f=np.zeros([6,1])
    f[[0,2,4]]=fx
    f[[1,3,5]]=fy
    return f
def boundary_forces_quad(nodes,elements,t,traction_element,psi = 0, eta = 0):
    x1,x2,x3,x4=nodes[elements[traction_element]][:,0]
    y1,y2,y3,y4=nodes[elements[traction_element]][:,1]
    if psi==1:
        vector = np.array([x3 - x2, y3 - y2])
        J=np.linalg.norm(vector)/2
    if psi==-1:
        vector = np.array([x4 - x1, y4 - y1])
        J=np.linalg.norm(vector)/2
    if eta==1:
        vector = np.array([x3 - x4, y3 - y4])
        J=np.linalg.norm(vector)/2
    if eta==-1:
        vector = np.array([x1 - x2, y1 - y2])
        J=np.linalg.norm(vector)/2
    w=2
    N=np.array([[(1-psi)*(1-eta)*0.25],
                 [(1+psi)*(1-eta)*0.25],
                 [(1+psi)*(1+eta)*0.25],
                [(1-psi)*(1+eta)*0.25]])
    integral=w*J*N
    fx=t[0]*integral
    fy=t[1]*integral
    f=np.zeros([8,1])
    f[[0,2,4,6]]=fx
    f[[1,3,5,7]]=fy
    return f
def assemble_force(global_force,force,elements,traction_element):
    s = len(elements[traction_element])
    c = np.zeros([2*s])
    c[np.array(range(0,2*s,2))] = 2 * elements[traction_element]
    c[np.array(range(1,2*s,2))] = 2 * elements[traction_element]+ 1
    node = c.astype("int32")
    global_force[node,:] += force
    return global_force
def global_force(nodes,elements,traction_element_list,forces):
    gforce = np.zeros([2*len(nodes),1])
    for i in range(len(traction_element_list)):
        gforce = assemble_force(gforce,forces[i],elements,traction_element_list[i])
    return gforce