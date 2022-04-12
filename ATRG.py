# coding:utf-8

# An implementatio of ATRG based on D. Adachi, et al PRB (2020).

import numpy as np
import scipy as scipy
import scipy.linalg as linalg

def Trace_tensor(A):
    # contraction of a single Tensor with the perioic boundary condition
    Trace_A = np.trace(A,axis1=0,axis2=2)
    Trace_A = np.trace(Trace_A)
    return Trace_A

def Trace_tensor2(A1,A2,sd):
    # contraction of a single Tensor with the perioic boundary condition
    A_com = np.tensordot(np.tensordot(A1,np.diag(sd),axes=(2,0)),A2,axes=(2,0))
    Trace_A = np.trace(A_com,axis1=0,axis2=2)
    Trace_A = np.trace(Trace_A)
    return Trace_A

def initialize_A(T):
    # Make initial tensor of square lattice Ising model at a temperature T
    A =np.empty((2,2,2,2))
    
    for i in range(0,2):
        si = (i - 0.5) * 2
        for j in range(0,2):
            sj = (j - 0.5) * 2
            for k in range(0,2):
                sk = (k - 0.5) * 2
                for l in range(0,2):
                    sl = (l - 0.5) * 2

                    A[i,j,k,l] = np.exp((si*sj + sj*sk + sk*sl + sl*si)/T)
                    
    ## normalization of tensor
    factor = Trace_tensor(A)

    
    A /= factor
    
    return A,factor

def SVD_first(A,D):
    ## sigular value decomposition and truncation of A
    A_mat1=np.reshape(A,(A.shape[0]*A.shape[1],A.shape[2]*A.shape[3]))
        
    U,s,VT = linalg.svd(A_mat1)

    ## truncate singular values at D_cut
    D_cut = np.min((s.size,D))
    
    U_t = U[:,0:D_cut].reshape((A.shape[0],A.shape[1],D_cut))
    VT_t = VT[0:D_cut,:].reshape((D_cut,A.shape[2],A.shape[3]))
    s_t = s[0:D_cut]
    return U_t, VT_t, s_t

def SVD_main(Tu,Td,D):    
    T_mat = np.tensordot(Tu,Td,axes=(2,1)).transpose(2,0,1,3).reshape(Td.shape[0]*Tu.shape[0],Tu.shape[1]*Td.shape[2])
    U,s,VT = linalg.svd(T_mat)
    
    ## truncate singular values at D_cut
    D_cut = np.min((s.size,D))
    
    U_t = U[:,0:D_cut].reshape((Td.shape[0],Tu.shape[0],D_cut))
    VT_t = VT[0:D_cut,:].reshape((D_cut,Tu.shape[1],Td.shape[2]))
    s_t = s[0:D_cut]
    return U_t, VT_t, s_t

def SVD_projector(T1,T2,T3,T4,D):    
        
    T_mat = np.tensordot(np.tensordot(T1,T2,axes=(2,0)),np.tensordot(T3,T4,axes=(2,1)),axes=((1,2),(0,2))).reshape(T1.shape[0]*T2.shape[2],T3.shape[1]*T4.shape[2])
    U,s,VT = linalg.svd(T_mat)
    
    ## truncate singular values at D_cut
    D_cut = np.min((s.size,D))
    
    U_t = U[:,0:D_cut].reshape((T1.shape[0],T2.shape[2],D_cut))
    VT_t = VT[0:D_cut,:].reshape((D_cut,T3.shape[1],T4.shape[2]))
    s_t = s[0:D_cut]
    return U_t, VT_t, s_t

def make_next_A(G,H):
    QG,RG = linalg.qr(G.reshape(G.shape[0]*G.shape[1],G.shape[2]),mode="economic")
    QH,RH = linalg.qr(H.reshape(H.shape[0],H.shape[1]*H.shape[2]).T,mode="economic")

    U,s,VT = linalg.svd(np.tensordot(RG,RH,axes=(1,1)))

    A1 = np.dot(QG,U).reshape(G.shape[0],G.shape[1],G.shape[2]).transpose(1,0,2)
    A2 = np.dot(VT,QH.T).reshape(H.shape[0],H.shape[1],H.shape[2]).transpose(0,2,1)
    sd = s

    return A1,A2,sd
def Update_Atensor(A1,A2,sd,D):    

    A1_d = np.tensordot(A1,np.diag(sd),axes=(2,0))
    A2_d = np.tensordot(np.diag(sd),A2,axes=(1,0))

    U,VT,s = SVD_main(A2_d,A1_d,D)

    s_sq = np.sqrt(s)
    T1 = np.tensordot(np.diag(s_sq),VT,axes=(1,0))
    T2 = A2
    T3 = A1
    T4 = np.tensordot(U,np.diag(s_sq),axes=(2,0))

    U,VT,s = SVD_projector(T1,T2,T3,T4,D)

    s_sq = np.sqrt(s)
    G = np.tensordot(np.diag(s_sq),VT,axes=(1,0))
    H = np.tensordot(U,np.diag(s_sq),axes=(2,0)).transpose(0,2,1)

    A1_new,A2_new,sd_new = make_next_A(G,H)

    ## test
    #Tn = np.tensordot(G,H,axes=(2,0)).transpose(1,0,3,2)
    #A1_new,A2_new,sd_new = SVD_first(Tn,D)
    ## test end

    factor = Trace_tensor2(A1_new,A2_new,sd_new)

    sd_new /= factor
    return A1_new, A2_new,sd_new,factor

    
##### Main part of TRG ####
def ATRG_Square_Ising(T,D,TRG_steps):

    ##  Initialization ##
    A,factor = initialize_A(T)
    
    TRG_factors = [factor]
    
    ## TRG iteration ##
    for i_TRG in range(0,TRG_steps):

        if i_TRG ==0:
            A1,A2,sd = SVD_first(A,D)
            
        A1,A2,sd,factor = Update_Atensor(A1,A2,sd,D)
        TRG_factors.append(factor)

    ## End of TRG iteration

    #Calclation of free energy
    free_energy_density = 0.0
    for i_TRG in range(TRG_steps+1):
        free_energy_density +=  np.log(TRG_factors[i_TRG]) * 0.5**i_TRG
    ## note: we normalize A so that Trace(A) = 1.0
    free_energy_density = -T * 0.5 * (free_energy_density + np.log(Trace_tensor2(A1,A2,sd))) 
    #print("T, free_energy_density = "+repr(T)+" " +repr(free_energy_density))
    return free_energy_density

