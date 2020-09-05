from numba import jit
from faiss import IndexFlatL2
from numpy import array
from numba import cuda
import numpy as np
from numpy import argsort
import faiss

# pc is the dataset, which should be a point cloud
# k1 is the number of neighbors of neighborhood graph
# k2 is k-nearest neighbor used to construct denstiy map
#tau1 is bandiwidth used to construct density map

def feature(pc,k1,k2,tau1):
    f = approximate(pc, k2,tau1)
    f = f.astype(float)
    pc = pc.astype(float)
    sorted_idxs = np.argsort(f)
    f = f[sorted_idxs]
    pc = pc[sorted_idxs]
    lims, I = rips_graph(pc, k1)
    ddd,ddqq = Clustering(f,I, lims, 1)
    see = []
    for i in ddqq:
        if (i == np.array([-1,-1])).all():
            pass
        else:
            see.append(i)
    see = np.array(see)
    result = []
    for i in np.unique(see[:,0]):
        result.append([see[np.where(see[:,0] == i)[0]][0,0], max(see[np.where(see[:,0] == i)[0]][:,1]) ])
    result = np.array(result)
    pdpairs = result
    for key,value in ddd.items():
        ddd[key] = sorted_idxs[ddd[key]]
    return ddd,f[result], pc, f, I, pdpairs

def gradient(pc,f,I,pdpairs,destnum):
    oripd = f[pdpairs]
    sorted_idxs = np.argsort(oripd[:,0] - oripd[:,1])
    changing = sorted_idxs[:-destnum]
    changepairs = pdpairs[changing]
    grad = np.zeros(pc.shape)
    for i in changepairs:
        coeff = np.exp( -np.linalg.norm(pc[i[0]] - pc[I[i[0]]],axis = 1) ** 2 / tau1)
        direction = pc[i[0]] - pc[I[i[0]]]
        grad[I[i[0]]] =  multp(direction, coeff)
        coeff1 = -np.exp( -np.linalg.norm(pc[i[1]] - pc[I[i[1]]],axis = 1) ** 2 / tau1)
        direction1 = pc[i[1]] - pc[I[i[1]]]
        grad[I[i[1]]] =  multp(direction1, coeff1)
    return pc,oripd


def genpd(pc, k1,k2,tau1,thresh):
    f,I1 = approximate(pc, k2,tau1)
#         print(pc)
    f = f.astype(float)
    pc = pc.astype(float)
    sorted_idxs = np.argsort(f)
    f = f[sorted_idxs]
    pc = pc[sorted_idxs]

    lims, I = rips_graph(pc, k1)
#         print(I)
    ddd,ddqq = Clustering(f,I, lims, thresh)
    if thresh == 1:
        see = []
        for i in ddqq:
            if (i == np.array([-1,-1])).all():
                pass
            else:
                see.append(i)
        see = np.array(see)
        result = []
        for i in np.unique(see[:,0]):
            result.append([see[np.where(see[:,0] == i)[0]][0,0], max(see[np.where(see[:,0] == i)[0]][:,1]) ])
        result = np.array(result)
        for key,value in ddd.items():
            ddd[key] = sorted_idxs[ddd[key]]
        return ddd,f[result]
    else:
        for key,value in ddd.items():
            ddd[key] = sorted_idxs[ddd[key]]
        return ddd,np.array([[0,0]])


def major(pc, k1,k2,tau1, destnum, learning_rate, epoch_num):
    for io in range(epoch_num):
        print(io)
        f,I1 = approximate(pc, k2,tau1)
        f = f.astype(float)
        pc = pc.astype(float)
        sorted_idxs = np.argsort(f)
        I1 = I1[sorted_idxs]
        f = f[sorted_idxs]
        pc = pc[sorted_idxs]
        lims, I = rips_graph(pc, k1)
#             print(lims)
        ddd,ddqq = Clustering(f,I, lims, 1)
        see = []
        for i in ddqq:
            if (i == np.array([-1,-1])).all():
                pass
            else:
                see.append(i)
        see = np.array(see)
        result = []
        for i in np.unique(see[:,0]):
            result.append([see[np.where(see[:,0] == i)[0]][0,0], max(see[np.where(see[:,0] == i)[0]][:,1]) ])
        result = np.array(result)
        pdpairs = result
        oripd = f[result]
        sorted_idxs = np.argsort(oripd[:,0] - oripd[:,1])
        changing = sorted_idxs[:-destnum]
        nochanging = sorted_idxs[-destnum:-1]
        biggest = oripd[sorted_idxs[-1]]
        dest = np.array([biggest[0], biggest[1]])
        changepairs = pdpairs[changing]
        nochangepairs = pdpairs[nochanging]
#             print(oripd)
        for i in changepairs:
            coeff = np.sqrt(2)/len(changepairs) * np.exp( -np.linalg.norm(pc[i[0]] - pc[I1[i[0]]],axis = 1) ** 2 / tau1)
            direction = pc[i[0]] - pc[I1[i[0]]]
            pc[I1[i[0]]] = pc[I1[i[0]]] - learning_rate * multp(direction, coeff)
            coeff1 = -np.sqrt(2)/len(changepairs) * np.exp( -np.linalg.norm(pc[i[1]] - pc[I1[i[1]]],axis = 1) ** 2 / tau1)
            direction1 = pc[i[1]] - pc[I1[i[1]]]
            pc[I1[i[1]]] = pc[I1[i[1]]] - learning_rate * multp(direction1, coeff1)
        
        pd11 = f[changepairs]
        print( 'weaking dist: ' + str(np.sum(pd11[:,0] - pd11[:,1])/2)  ) 
        sal = 0
        for i in nochangepairs:
            dist = np.linalg.norm(f[i] - dest)   
#            print(dist)
            if dist == 0:
                pass
            else:
                coeff = 1/dist * (f[i[0]] - dest[0]) /tau1/len(nochangepairs) * np.exp( -np.linalg.norm(pc[i[0]] - pc[I1[i[0]]],axis = 1) ** 2 / tau1)
                direction = pc[i[0]] - pc[I1[i[0]]]
                pc[I1[i[0]]] = pc[I1[i[0]]] - learning_rate * multp(direction, coeff)
                coeff1 = 1/dist * (f[i[1]] - dest[1]) /tau1 /len(nochangepairs)* np.exp( -np.linalg.norm(pc[i[1]] - pc[I1[i[1]]],axis = 1) ** 2 / tau1)
                direction1 = pc[i[1]] - pc[I1[i[1]]]
                pc[I1[i[1]]] = pc[I1[i[1]]] - learning_rate * multp(direction1, coeff1)
            
                sal = sal + dist
        print( 'salient dist: ' + str(sal))
        
    return pc,f[result]

def loss( pd, destnum ):
    sorted_idxs = np.argsort(pd[:,0] - pd[:,1])
    pd = pd[sorted_idxs]
    

def multp(a,b):
    for i in range( a.shape[1] ):
        a[:,i] = a[:,i] * b
    return a
                
def approximate(pc, r,tau):

    pc = pc.astype('float32')
    size = len(pc)
    index = IndexFlatL2(len(pc[0]))
    index.add(pc)
    D,I = index.search(pc, r)
    result = np.sum(np.exp(-D/tau),axis = 1)/(r * tau)
    return result/ max(result),I #/ max(result)

@jit(nopython = True)
def find_entry_idx_by_point(entries, point_idx):
    for index, entry in entries.items():
        for i in entry:
            if i == point_idx:
                return np.int64(index)

@jit(nopython = True)
def Clustering(f,I, lims,tau):
    ggg =np.array([[-1,-1]])
    entries = {f.shape[0] - 1 : np.array([f.shape[0] - 1])}
    for i in np.arange(f.shape[0] - 2,-1,-1):
        nbr_idxs = I[i]
#         print(I[i])
        upper_star_idxs = nbr_idxs[nbr_idxs >= i]   
        if upper_star_idxs.size == 1:
            # i is a local maximum
            entries[i] = np.array([i])
        else:
#             print(upper_star_idxs)
            g_i = np.max(upper_star_idxs)
            entry_idx = find_entry_idx_by_point(entries, g_i)
            entries[entry_idx] = np.append(entries[entry_idx], i)
            entries,kkk = merge(f, entries, i, upper_star_idxs, tau)
            if len(kkk)>1:
                ggg = np.append(ggg, kkk,axis = 0)
    return entries,ggg

@jit(nopython = True)
def merge(f, entries, i,upper_star_idxs,tau):
    ggg = np.array([[-1,-1]])
    main_entry_idx = find_entry_idx_by_point(entries, i)
    
    for j in range(len(upper_star_idxs)):
        star_idx = find_entry_idx_by_point(entries, upper_star_idxs[j])
        if j == 0:
            e_up = star_idx
        elif f[np.int64(star_idx)] > f[np.int64(e_up)]:
            e_up = star_idx
            
    for j in upper_star_idxs:
        entry_idx = find_entry_idx_by_point(entries, j)

        if (e_up != entry_idx)and(f[np.int64(entry_idx)] - f[np.int64(i)] < tau):
#             if (f[np.int64(entry_idx)] - f[np.int64(i)] < tau):
            ggg = np.append(ggg, np.array([[int(entry_idx),int(i)]]) ,axis = 0)
            entries[e_up] = np.append(entries[e_up], entries[entry_idx])
            entries.pop(entry_idx)            
     
    return entries,ggg



def rips_graph(point_cloud, k):
    point_cloud = point_cloud.astype('float32')
    _, dim = point_cloud.shape
    cpuindex = IndexFlatL2(dim)
    cpuindex.add(point_cloud)

    return cpuindex.search(point_cloud, k)