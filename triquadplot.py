import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import meshio
def plotting_mesh(nodes,elements):
    plt.scatter(nodes[:,0],nodes[:,1], label="original nodes")
    for i,node in enumerate(nodes):
        plt.annotate(i,(node[0],node[1]))
    for i,element in enumerate(elements):
        x1=nodes[element,0]
        y1=nodes[element,1]
        plt.fill(x1,y1,edgecolor="black",fill=False)
        center_x1=np.mean(x1)
        center_y1=np.mean(y1)
        plt.annotate(i,(center_x1,center_y1))
        center_x1=np.mean(np.concatenate((x1,[x1[0]])))
        center_y1=np.mean(np.concatenate((y1,[y1[0]])))
        plt.annotate("*",(center_x1,center_y1))
    plt.legend()
    plt.show()
    return None
def plotting(nodes,elements,U):
    plt.scatter(nodes[:,0],nodes[:,1])
    displaced_nodes =nodes.astype(float).copy()
    displaced_nodes[:,0] += U[::2,0]
    displaced_nodes[:,1] += U[1::2,0]  
    plt.scatter(displaced_nodes[:,0],displaced_nodes[:,1], label='Displaced nodes')
    for i,node in enumerate(nodes):
        plt.annotate(i,(node[0],node[1]))
    for i,element in enumerate(elements):
        x= displaced_nodes[element,0]
        y = displaced_nodes[element,1]
        x1=nodes[element,0]
        y1=nodes[element,1]
        plt.fill(x1,y1,edgecolor="black",fill=False)
        plt.fill(x,y,edgecolor="red",fill=False)
        # center_x=np.mean(x)
        # center_y=np.mean(y)
        # center_x1=np.mean(x1)
        # center_y1=np.mean(y1)
        # plt.annotate(i,(center_x,center_y))
        # plt.annotate(i,(center_x1,center_y1))
    plt.legend()
    plt.show()
    return None
def display_tri(nodes,U,elements,value):
    
    displaced_nodes =nodes.astype(float).copy()
    displaced_nodes[:,0] += U[::2,0]
    displaced_nodes[:,1] += U[1::2,0]
    cmap = cm.get_cmap("jet")
    norm = plt.Normalize(value.min(),value.max())
    colors = cmap(norm(value))
    for i in range(len(elements)):
        co = displaced_nodes[elements[i]]
        x = co[:,0]
        y = co[:,1]
        plt.fill(x,y,color = colors[i])
    sc = cm.ScalarMappable(norm=norm,cmap=cmap)
    plt.colorbar(sc)
    return norm
def display_quad(nodes,elements,ax,value):   
    ax.set_aspect('equal')
    ax.scatter(nodes[:, 0], nodes[:, 1])
    vmin = min(value.flatten())
    vmax = max(value.flatten())
    for i in range(len(elements)):
        co = nodes[elements[i]]
        corner = value[i]
        a = np.repeat([np.linspace(0, 1, 30)], 30, axis=0)
        b = np.repeat(np.linspace(0, 1, 30), 30)
        a = np.hstack(a)
        n = len(a)
        inter = (1 - a) * (1 - b) * corner[0] + a * (1 - b) * corner[1] + b * (1 - a) * corner[3] + a * b * corner[2]
        ref = np.transpose(np.concatenate(([a], [b]), axis=0))
        psi = np.reshape(ref[:, 0] * 2 - 1, (n, 1))
        eta = np.reshape(ref[:, 1] * 2 - 1, (n, 1))
        N1 = (1 - psi) * (1 - eta) / 4
        N2 = (1 + psi) * (1 - eta) / 4
        N3 = (1 + psi) * (1 + eta) / 4
        N4 = (1 - psi) * (1 + eta) / 4
        N = np.concatenate((N1, N2, N3, N4), axis=1)
        xy = np.dot(N, co)
        plt.scatter(xy[:, 0], xy[:, 1], c=inter, cmap="jet",vmin=vmin,vmax=vmax)
    plt.colorbar()
    return None