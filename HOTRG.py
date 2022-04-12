# coding:utf-8

# Higher-order Tensor Renormalization Groupe (HOTRG)
# based on Xie et al, PRB (2012)
# Bond-weighed HOTRG (BHOTRG)
# based on D. Adachi, et al., PRB (2021).

import numpy as np
import scipy as scipy
import scipy.linalg as linalg
import TRG

def Trace_tensor(A):
    # contraction of a single Tensor with the perioic boundary condition
    Trace_A = np.trace(A,axis1=0,axis2=2)
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


def HOSVD(A,dt1,dt2,D,kp):
    ## sigular value decomposition and truncation of A
    dt1_sq = np.sqrt(dt1)

    #A_mod = np.tensordot(np.tensordot(np.diag(dt1_sq),A,axes=(1,0)),np.diag(dt1_sq),axes=(2,0)).transpose(0,1,3,2)
    A_mod = np.tensordot(A,np.diag(dt1_sq),axes=(2,0)).transpose(0,1,3,2)
    A_up = np.tensordot(A_mod,np.diag(dt2),axes=(3,0))
    A_down = A_mod

    ##following definition is exactly same with k = -k
    #A_mod = np.tensordot(np.tensordot(np.diag(dt1),A,axes=(1,0)),np.diag(dt1_sq),axes=(2,0)).transpose(0,1,3,2)
    #A_up = np.tensordot(np.diag(dt2),np.tensordot(A_mod,np.diag(dt2),axes=(3,0)),axes=(1,1)).transpose(1,0,2,3)
    #A_down = np.tensordot(A_mod,np.diag(dt2),axes=(3,0))
    
    AAu = np.tensordot(A_up.conj(),A_up,axes=((0,1),(0,1)))
    AAd = np.tensordot(A_down.conj(),A_down,axes=((0,3),(0,3)))

    A4 = np.tensordot(AAu,AAd,axes=((1,3),(0,2))).transpose(0,2,1,3).reshape(A.shape[2]**2,A.shape[2]**2)
        
    U,s,VT = linalg.svd(A4)

    ## truncate singular values at D_cut
    D_cut = np.min((s.size,D))
    
    U = np.dot(U[:,:D_cut],np.diag(s[0:D_cut]**(kp))).reshape(A.shape[2],A.shape[2],D_cut)
    
    return U,s[:D_cut]**(-2*kp)

def HOSVD_New(A,dt1,dt2,D,kp):
    ## This routine does not work efficiently
    ## sigular value decomposition and truncation of A
    dt1_sq = np.sqrt(dt1)

    #A_mod = np.tensordot(np.tensordot(np.diag(dt1_sq),A,axes=(1,0)),np.diag(dt1_sq),axes=(2,0)).transpose(0,1,3,2)
    A_mod = np.tensordot(A,np.diag(dt1_sq),axes=(2,0)).transpose(0,1,3,2)
    A_up = np.tensordot(A_mod,np.diag(dt2),axes=(3,0))
    A_down = A_mod

    ##following definition is exactly same with k = -k
    #A_mod = np.tensordot(np.tensordot(np.diag(dt1),A,axes=(1,0)),np.diag(dt1_sq),axes=(2,0)).transpose(0,1,3,2)
    #A_up = np.tensordot(np.diag(dt2),np.tensordot(A_mod,np.diag(dt2),axes=(3,0)),axes=(1,1)).transpose(1,0,2,3)
    #A_down = np.tensordot(A_mod,np.diag(dt2),axes=(3,0))
    
    AAu = np.tensordot(A_up.conj(),A_up,axes=((0,1),(0,1)))
    AAd = np.tensordot(A_down.conj(),A_down,axes=((0,3),(0,3)))

    A4 = np.tensordot(AAu,AAd,axes=((1,3),(0,2))).transpose(0,2,1,3).reshape(A.shape[2]**2,A.shape[2]**2)
        
    U,s,VT = linalg.svd(A4)

    ## truncate singular values at D_cut
    D_cut = np.min((s.size,D))

    ## redefine projector and s
    #AA_U = np.tensordot(U[:,:D_cut],np.tensordot(A_up,A_down,axes=((1,3),(3,1))).transpose(1,3,0,2).reshape(A.shape[2]**2,A.shape[0]**2),axes=(0,0))
    #U_2,s_2,VT_2 = linalg.svd(AA_U)
    #Un = np.dot(np.dot(U[:,:D_cut],U_2),np.diag(s_2**kp)).reshape(A.shape[2],A.shape[2],D_cut)


    dt1_sq = np.sqrt(dt1)
    A_mod = np.tensordot(np.tensordot(np.diag(dt1_sq),A,axes=(1,0)),np.diag(dt1_sq),axes=(2,0)).transpose(0,1,3,2)    
    A_mod_u = np.tensordot(np.tensordot(np.tensordot(np.diag(dt1_sq),A,axes=(1,0)),np.diag(dt1_sq),axes=(2,0)),np.diag(dt2),axes=(2,0))
    
    #A_new = Combine_two(A_mod_u,A_mod,U,U.conj())
    P = U[:,:D_cut].reshape(A.shape[2],A.shape[2],D_cut)
    A_new = Combine_two_mem(A_mod_u,A_mod,P,P.conj()).transpose(3,0,1,2)

    Un,sn,VTn = linalg.svd(A_new.reshape(A_new.shape[0]*A_new.shape[1],A_new.shape[2]*A_new.shape[3]))

    M = np.dot(np.diag(sn),VTn).reshape(len(sn),A_new.shape[2],A_new.shape[3]).transpose(0,2,1).reshape(len(sn)*A_new.shape[3],A_new.shape[2])
    U3,s3,VT3 = linalg.svd(M)

        
    Un = np.dot(np.tensordot(U[:,:D_cut],VT3,axes=(1,1)),np.diag(s3**kp)).reshape(A.shape[2],A.shape[2],D_cut)

    
    return Un,s3**(-2*kp)


def Update_Atensor(A,dt1,dt2,D,kp):    
    
    U,dt1_new = HOSVD(A,dt1,dt2,D,kp)
    #U,dt1_new = HOSVD_New(A,dt1,dt2,D,kp)

    dt1_sq = np.sqrt(dt1)
    A_mod = np.tensordot(np.tensordot(np.diag(dt1_sq),A,axes=(1,0)),np.diag(dt1_sq),axes=(2,0)).transpose(0,1,3,2)    
    A_mod_u = np.tensordot(np.tensordot(np.tensordot(np.diag(dt1_sq),A,axes=(1,0)),np.diag(dt1_sq),axes=(2,0)),np.diag(dt2),axes=(2,0))
    
    #A_new = Combine_two(A_mod_u,A_mod,U,U.conj())
    A_new = Combine_two_mem(A_mod_u,A_mod,U,U.conj())
    factor = TRG.Trace_tensor(A_new,dt2,dt1_new)

    factor_diag = np.max(dt1_new)
    factor_tensor = factor/factor_diag
    
    A_new /= factor_tensor
    dt1_new /= factor_diag


    return A_new,dt1_new,factor

def Combine_two_mem(A1,A2,P1,P2):
    D_in = A1.shape[3]
    A = np.zeros((A1.shape[1],P1.shape[2],P2.shape[2],A2.shape[3]))

    for i in range(D_in):    
        A1P = np.tensordot(A1[:,:,:,i],P1,axes=(2,0))
        A2P = np.tensordot(P2,A2[:,i,:,:],axes=(1,0))
        A += np.tensordot(A1P,A2P,axes=((0,2),(0,2)))
    return A.transpose(0,1,3,2)

def Combine_two(A1,A2,P1,P2):
    
    A1P = np.tensordot(A1,P1,axes=(2,0))
    A2P = np.tensordot(P2,A2,axes=(1,0))
    A = np.tensordot(A1P,A2P,axes=((0,2,3),(0,2,3))).transpose(0,1,3,2)
    return A

##### Main part of TRG ####
def TRG_Square_Ising(T,D,kp,TRG_steps):

    ##  Initialization ##
    A,dt1,dt2,factor = TRG.initialize_A(T)
    entropy1,entropy2 = TRG.Calc_Entropy(A,dt1,dt2)
    print("## Entroypy: "+repr(0)+" "+repr(entropy1)+" " + repr(entropy2))
    
    TRG_factors = [factor]
    
    ## TRG iteration ##
    for i_TRG in range(0,TRG_steps):

        A, dt_new ,factor = Update_Atensor(A,dt1,dt2,D,kp)

        dt1 = dt2.copy()
        dt2 = dt_new
        entropy1,entropy2 = TRG.Calc_Entropy(A,dt1,dt2)
        print("## Entroypy: "+repr(0)+" "+repr(entropy1)+" " + repr(entropy2))
        TRG_factors.append(factor)

    ## End of TRG iteration

    #Calclation of free energy
    free_energy_density = 0.0
    for i_TRG in range(TRG_steps+1):
        free_energy_density +=  np.log(TRG_factors[i_TRG]) * 0.5**i_TRG
    ## note: we normalize A so that Trace(A) = 1.0
    free_energy_density = -T * 0.5 * (free_energy_density + 0.5**TRG_steps*np.log(TRG.Trace_tensor(A,dt1,dt2))) 
    #print("T, free_energy_density = "+repr(T)+" " +repr(free_energy_density))
    return free_energy_density

