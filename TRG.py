# coding:utf-8

# Tensor Renormalization Group (TRG)
# based on M. Levin and C. P. Nave, PRL 99 120601 (2007)
# and X.-C. Gu, M. Levin, and X.-G. Wen, PRB 78, 205116(2008).
# mean-findl SRG, based on H.H. Zhao et al., PRB (2010)
# and implementation of Bond-weighted TRG (BTRG) based on
# D. Adachi et al, PRB (2021).

import numpy as np
import scipy as scipy
import scipy.linalg as linalg

def Trace_tensor(A,dt1,dt2):
    # contraction of a single Tensor with the perioic boundary condition
    A_2dt = np.tensordot(np.tensordot(np.diag(dt1),A,axes=(1,0)),np.diag(dt2),axes=(3,0))
    Trace_A = np.trace(A_2dt,axis1=0,axis2=2)
    Trace_A = np.trace(Trace_A)
    return Trace_A

def Trace_tensor_no_dt(A):
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
    dt1 = np.ones(2,dtype=float)
    dt2 = np.ones(2,dtype=float)
    factor = Trace_tensor(A,dt1,dt2)

    factor_diag = 0.5 * (np.max(dt1)+np.max(dt2))
    factor_tensor = factor/factor_diag**2
    
    A /= factor_tensor
    dt1 /= factor_diag
    dt2 /= factor_diag
    
    return A,dt1,dt2,factor

def SVD_type1(A,D,kp):
    ## sigular value decomposition and truncation of A
    A_mat1=np.reshape(A,(A.shape[0]*A.shape[1],A.shape[2]*A.shape[3]))
        
    U,s,VT = linalg.svd(A_mat1)

    ## truncate singular values at D_cut
    D_cut = np.min((s.size,D))
    s_t = s[0:D_cut]**kp
    
    S3 = np.dot(U[:,0:D_cut],np.diag(s_t))
    S1 = np.dot(np.diag(s_t),VT[0:D_cut,:])
    S3 = np.reshape(S3,(A.shape[0],A.shape[1],D_cut))
    S1 = np.reshape(S1,(D_cut,A.shape[2],A.shape[3]))

    return S1,S3,s[0:D_cut]/s_t**2
def SVD_type2(A,D,kp):
    ## sigular value decomposition and truncation of A
    A_mat2 = np.transpose(A,(0,3,1,2))
    A_mat2 = np.reshape(A_mat2,(A.shape[0]*A.shape[3],A.shape[1]*A.shape[2]))
        
    U,s,VT = linalg.svd(A_mat2)
    
    ## truncate singular values at D_cut
    D_cut = np.min((s.size,D))
    s_t = s[0:D_cut]**kp

    S2 = np.dot(U[:,0:D_cut],np.diag(s_t))
    S4 = np.dot(np.diag(s_t),VT[0:D_cut,:])
    S2 = np.reshape(S2,(A.shape[0],A.shape[3],D_cut))
    S4 = np.reshape(S4,(D_cut,A.shape[1],A.shape[2]))

            
    return S2,S4,s[0:D_cut]/s_t**2



def SVD_meanfield(A,dt1,dt2,D):
    ## In the case of square lattice, we cannot assume symmetry
    if dt1 is None:
        dt1_sq = np.ones(A.shape[0])
    else:
        dt1_sq = np.sqrt(dt1)
    if dt2 is None:
        dt2_sq = np.ones(A.shape[1])
    else:
        dt2_sq = np.sqrt(dt2)
        
    ## sigular value decomposition and truncation of A
    
    A_temp=np.tensordot(np.diag(dt1_sq),np.tensordot(np.diag(dt2_sq),np.tensordot(np.diag(dt1_sq),np.tensordot(np.diag(dt2_sq),A,axes=(1,3)),axes=(1,3)),axes=(1,3)),axes=(1,3))
    A_mat1 = np.reshape(A_temp,(A.shape[0]*A.shape[1],A.shape[2]*A.shape[3]))
    A_mat2 = np.reshape(A_temp.transpose(0,3,1,2),(A.shape[0]*A.shape[3],A.shape[1]*A.shape[2]))
        
    U1,s1,VT1 = linalg.svd(A_mat1)
    U2,s2,VT2 = linalg.svd(A_mat2)

    ## truncate singular values at D_cut
    D_cut = np.min((s1.size,D))

    s_t1 = s1[0:D_cut]
    s_t2 = s2[0:D_cut]
    
    if D_cut != s1.size:
        #truncation_error  = np.sum(s[D_cut:])/np.sqrt(np.sum(s**2))
        truncation_error1  = np.sum(s1[D_cut:])/np.sum(s1)
        truncation_error2  = np.sum(s2[D_cut:])/np.sum(s2)
    else:
        truncation_error1 = 0.0        
        truncation_error2 = 0.0        
    #print("truncation error = "+repr(truncation_error))


    dt1_sq_inv = np.zeros(len(dt1_sq))
    dt2_sq_inv = np.zeros(len(dt2_sq))
    for i in range(len(dt1_sq)):
        if dt1_sq[i] > 1e-16:
            dt1_sq_inv[i] = 1.0/dt1_sq[i]
    for i in range(len(dt2_sq)):
        if dt2_sq[i] > 1e-16:
            dt2_sq_inv[i] = 1.0/dt2_sq[i]

    
    S3 = np.tensordot(np.diag(dt1_sq_inv),np.tensordot(np.diag(dt2_sq_inv),np.dot(U1[:,0:D_cut],np.diag(np.sqrt(s_t1))).reshape(A.shape[0],A.shape[1],D_cut),axes=(1,1)),axes=(1,1))
    S1 = np.tensordot(np.tensordot(np.dot(np.diag(np.sqrt(s_t1)),VT1[0:D_cut,:]).reshape(D_cut,A.shape[2],A.shape[3]),np.diag(dt1_sq_inv),axes=(1,0)),np.diag(dt2_sq_inv),axes=(1,0))

    S2 = np.tensordot(np.diag(dt1_sq_inv),np.tensordot(np.diag(dt2_sq_inv),np.dot(U2[:,0:D_cut],np.diag(np.sqrt(s_t2))).reshape(A.shape[0],A.shape[3],D_cut),axes=(1,1)),axes=(1,1))
    S4 = np.tensordot(np.tensordot(np.dot(np.diag(np.sqrt(s_t2)),VT2[0:D_cut,:]).reshape(D_cut,A.shape[1],A.shape[2]),np.diag(dt2_sq_inv),axes=(1,0)),np.diag(dt1_sq_inv),axes=(1,0))
    

    return S1,S2,S3,S4,s_t1,s_t2,truncation_error1,truncation_error2


def Calc_Entropy(A,dt1,dt2):
    dt1_sq = np.sqrt(dt1)
    dt2_sq = np.sqrt(dt2)

    A_mod = np.tensordot(np.tensordot(np.tensordot(np.tensordot(A,np.diag(dt1_sq),axes=(0,0)),np.diag(dt2_sq),axes=(0,0)),np.diag(dt1_sq),axes=(0,0)),np.diag(dt2_sq),axes=(0,0))

    s1 = linalg.svdvals(A_mod.reshape(A.shape[0]*A.shape[1],A.shape[2]*A.shape[3]))
    s2 = linalg.svdvals(A_mod.transpose(3,0,1,2).reshape(A.shape[3]*A.shape[0],A.shape[1]*A.shape[2]))

    s1 /= np.sum(s1)
    s2 /= np.sum(s2)

    entropy1 = - np.dot(s1,np.log(s1))
    entropy2 = - np.dot(s2,np.log(s2))

    ## scaling dimension
    A_mat = np.trace(A_mod,axis1=0,axis2=2)
    lam = linalg.eigvals(A_mat)
    
    return entropy1,entropy2,lam
def Update_Atensor(A,dt1,dt2,D,kp):    

    S1,S3,dt1_new = SVD_type1(A,D,kp)
    S2,S4,dt2_new = SVD_type2(A,D,kp)



    A = Combine_four_S(S1,S2,S3,S4,dt1,dt2)
    factor = Trace_tensor(A,dt1_new,dt2_new)

    factor_diag = 0.5 * (np.max(dt1_new)+np.max(dt2_new))
    factor_tensor = factor/factor_diag**2

    #print("factors = "+repr(factor) + " " +repr(factor_tensor) + "  " +repr(factor_diag))
    #print("maxval = "+repr(np.max(np.abs(A))))
    #print("dt1 = "+repr(dt1_new))
    #print("dt2 = "+repr(dt2_new))
    A /= factor_tensor
    dt1_new /= factor_diag
    dt2_new /= factor_diag

    return A, dt1_new,dt2_new,factor


def Update_Atensor_meanfield(A,dt1,dt2,D,fixed_iteration,itr,epsilon):    

    if dt1 is None:
        dt1_new = None
    else:
        dt1_new = dt1/dt1[0]

    if dt2 is None:
        dt2_new = None
    else:
        dt2_new = dt2/dt2[0]
        

    S1,S2,S3,S4,dt1_new,dt2_new,truncation_error1,truncation_error2 = SVD_meanfield(A,dt1_new,dt2_new,D)

    diff_truncation = 1.0
    D_edge = A.shape[0]

    truncation_error = (truncation_error1 + truncation_error2) * 0.5
    
    if fixed_iteration:
        for i in range(itr):
            dt1_new /= dt1_new[0]
            dt2_new /= dt2_new[0]
            S1,S2,S3,S4,dt1_new,dt2_new,truncation_error1,truncation_error2 = SVD_meanfield(A,dt1_new[:D_edge],dt2_new[:D_edge],D)
            truncation_error = (truncation_error1 + truncation_error2) * 0.5
    else:
        while truncation_error > epsilon and diff_truncation > epsilon:
            dt1_new /= dt1_new[0]
            dt2_new /= dt2_new[0]
            S1,S2,S3,S4,dt1_new,dt2_new,truncation_error1,truncation_error2 = SVD_meanfield(A,dt1_new[:D_edge],dt2_new[:D_edge],D)

            truncation_error_new = (truncation_error1 + truncation_error2) * 0.5

            diff_truncation = abs((truncation_error_new - truncation_error)/truncation_error)
            truncation_error = truncation_error_new


    A = Combine_four_S_no_dt(S1,S2,S3,S4)
    factor = Trace_tensor_no_dt(A)

    A /= factor
    return A, dt1_new/dt1_new[0], dt2_new/dt2_new[0],factor,truncation_error

def Combine_four_S(S1,S2,S3,S4,dt1,dt2):
    S12 = np.tensordot(np.tensordot(S1,np.tensordot(np.diag(dt1),S2,axes=(1,0)),axes=(1,0)),np.diag(dt2),axes=(1,0))
    S43 = np.tensordot(np.tensordot(S4,np.tensordot(np.diag(dt1),S3,axes=(1,0)),axes=(2,0)),np.diag(dt2),axes=(2,0))

    A = np.tensordot(S12,S43,axes=([1,3],[3,1])).transpose(0,1,3,2)
    return A

def Combine_four_S_no_dt(S1,S2,S3,S4):
    S12 = np.tensordot(S1,S2,axes=(1,0))
    S43 = np.tensordot(S4,S3,axes=(2,0))

    A = np.tensordot(S12,S43,axes=([1,2],[1,2])).transpose(0,1,3,2)
    return A

##### Main part of TRG ####
def TRG_Square_Ising(T,D,kp,TRG_steps):

    ##  Initialization ##
    A,dt1,dt2,factor = initialize_A(T)

    entropy1,entropy2,lam = Calc_Entropy(A,dt1,dt2)
    print("## Entroypy: "+repr(0)+" "+repr(entropy1)+" " + repr(entropy2))
    #print("## lam: "+repr(0)+" "+" ".join(map(str, np.log(lam.real)-np.log(lam[0].real))))
    print("## lam: "+repr(0)+" "+" ".join(map(str, lam)))
    TRG_factors = [factor]
    
    ## TRG iteration ##
    for i_TRG in range(0,TRG_steps):

        A, dt1,dt2,factor = Update_Atensor(A,dt1,dt2,D,kp)
        TRG_factors.append(factor)
    
        entropy1,entropy2,lam = Calc_Entropy(A,dt1,dt2)
        print("## Entroypy: "+repr(i_TRG+1)+" "+repr(entropy1)+" " + repr(entropy2))
        #print("## lam: "+repr(i_TRG+1)+" "+" ".join(map(str, np.log(lam.real)-np.log(lam[0].real))))
        print("## lam: "+repr(i_TRG+1)+" "+" ".join(map(str, lam)))
    ## End of TRG iteration

    #Calclation of free energy
    free_energy_density = 0.0
    for i_TRG in range(TRG_steps+1):
        free_energy_density +=  np.log(TRG_factors[i_TRG]) * 0.5**i_TRG
    ## note: we normalize A so that Trace(A) = 1.0
    free_energy_density = -T * 0.5 * (free_energy_density + 0.5**TRG_steps*np.log(Trace_tensor(A,dt1,dt2))) 
    #print("T, free_energy_density = "+repr(T)+" " +repr(free_energy_density))
    return free_energy_density


def SRG_Square_Ising(T,D,TRG_steps,fixed_iteration,itr,epsilon,use_previous_singular_value):

    ##  Initialization ##
    A,dt1,dt2,factor = initialize_A(T)

    TRG_factors = [factor]
    
    ## TRG iteration ##
    for i_TRG in range(0,TRG_steps):
        if use_previous_singular_value:
            A, dt1,dt2,factor,truncation_error = Update_Atensor_meanfield(A,dt1,dt2,D,fixed_iteration,itr,epsilon)
        else:
            A, dt1,dt2,factor,truncation_error = Update_Atensor_meanfield(A,None,None,D,fixed_iteration,itr,epsilon)
            
        TRG_factors.append(factor)
    ## End of TRG iteration

    #Calclation of free energy
    free_energy_density = 0.0
    for i_TRG in range(TRG_steps+1):
        free_energy_density +=  np.log(TRG_factors[i_TRG]) * 0.5**i_TRG
    ## note: we normalize A so that Trace(A) = 1.0
    free_energy_density = -T * 0.5 * (free_energy_density + 0.5**TRG_steps*np.log(Trace_tensor_no_dt(A))) 
    #print("T, free_energy_density = "+repr(T)+" " +repr(free_energy_density))
    return free_energy_density
