import numpy as np
from numpy.polynomial.legendre import leggauss  
def element_strain(nodes,elements,v,E,U):
    if len(elements[0])==3:
        e=np.zeros([3,len(elements)])
        s=np.zeros([3,len(elements)])
        for i in range(len(elements)):
            x1, x2, x3 = nodes[elements][i][:, 0]
            y1, y2, y3 = nodes[elements][i][:, 1]
            matrix= np.array([[1,x1,y1],[1,x2,y2],[1,x3,y3]])
            A = np.linalg.det(matrix) / 2
            B = np.array([[y2-y3,y3-y1,y1-y2],
                        [x3-x2,x1-x3,x2-x1]])*(1/(2*A))
            B1=np.zeros([3,6])
            B1[0, [0, 2, 4]] = B[0]
            B1[1, [1, 3, 5]] = B[1]
            B1[2, [0, 2, 4]] = B[1]
            B1[2, [1, 3 ,5]] = B[0] 
    
            c = np.zeros([6,1])
            c[[0,2,4],0] = elements[i]*2
            c[[1,3,5],0] = elements[i]*2 + 1
            c=c.astype(int).flatten()
            u = U[c]
            u = np.reshape(u, (-1, 1))  # reshape u to be a column vector
            e[:, i] = np.dot(B1, u).flatten() 
            
            D=np.array([[1,v,0],[v,1,0],[0,0,(1-v)/2]])*(E/(1-v**2))
            # D=np.array([[1-v,v,0],[v,1-v,0],[0,0,(1-2*v)/2]])*(E/((1+v)*(1-2*v)))
            s[:,i]+=np.dot(D,np.dot(B1,u)).flatten()
    else:
        e=[]
        s=[]
        def gauss_quadrature(num_points):
            psi_values, weights_psi = leggauss(num_points)
            eta_values, weights_eta = leggauss(num_points)
            return psi_values, eta_values, weights_psi,weights_eta
        num_points=2
        psi_values, eta_values,weights_psi,weights_eta=gauss_quadrature(num_points)
        for i in range(len(elements)): 
            e1 = np.zeros([3, 4])
            s1 = np.zeros([3, 4]) 
            x1,x2,x3,x4=nodes[elements][i][:,0]
            y1,y2,y3,y4=nodes[elements][i][:,1]
            points = [[-1,-1],[1,-1],[1,1],[-1,1]]
            for j in range(4):
                    eta = points[j][1]
                    psi = points[j][0]
                    B1=np.zeros([3,8])
                    BN=np.array([[eta-1,1-eta,1+eta,-1-eta],
                                [psi-1,-psi-1,1+psi,1-psi]])*0.25
                    a=np.array([[x1,y1],[x2,y2],[x3,y3],[x4,y4]])

                    J=np.dot(BN,a)
                    dN = np.dot(np.linalg.inv(J),BN)
                    # print(dN)
                    B1[0, [0, 2, 4,6]] = dN[0]
                    B1[1, [1, 3, 5,7]] = dN[1]
                    B1[2, [0, 2, 4,6]] = dN[1]
                    B1[2, [1, 3 ,5,7]] = dN[0] 
                    a = elements[i]*2
                    b = elements[i]*2 + 1
                    c = np.zeros([8,1])
                    c[[0,2,4,6],0] = a
                    c[[1,3,5,7],0] = b
                    c=c.astype(int).flatten()
                    u = U[c]
                    # print(np.dot(B1, u))
                    e1[:, j] = np.dot(B1, u)[:,0]
                    D=np.array([[1,v,0],[v,1,0],[0,0,(1-v)/2]])*(E/(1-v**2))
                    s1[:,j]=np.dot(D,np.dot(B1,u))[:,0]
            s.append(s1)
            e.append(e1)
    return  np.array(e),np.array(s)