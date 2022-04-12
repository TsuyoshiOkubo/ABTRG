# coding:utf-8

# Tensor Renormalization Groupe for the honeycomb lattice
# based on M. Levin and C. P. Nave, PRL 99 120601 (2007)
# and mean-filed SRG based on H.H. Zhao et al., PRB (2010).

import numpy as np
import scipy as scipy
import scipy.linalg as linalg

def Trace_tensor(A,B,dt0,dt1,dt2):
    # contraction of a six Tensor with the central symmetric perioic boundary condition
    A_dt = np.tensordot(np.tensordot(np.tensordot(A,np.diag(dt0),axes=(0,0)),np.diag(dt1),axes=(0,0)),np.diag(dt2),axes=(0,0))
    AB_0= np.tensordot(A_dt,B,axes=(0,0))
    AB_1= np.tensordot(A_dt,B,axes=(1,1))
    AB_2= np.tensordot(A_dt,B,axes=(2,2))

    Trace_AB = np.tensordot(np.tensordot(AB_0,AB_1,axes=((1,2),(3,1))),AB_2,axes=((0,1,2,3),(3,1,2,0)))
    return Trace_AB

def Trace_tensor_no_dt(A,B):
    # contraction of a six Tensor with the central symmetric perioic boundary condition
    AB_0= np.tensordot(A,B,axes=(0,0))
    AB_1= np.tensordot(A,B,axes=(1,1))
    AB_2= np.tensordot(A,B,axes=(2,2))

    Trace_AB = np.tensordot(np.tensordot(AB_0,AB_1,axes=((1,2),(3,1))),AB_2,axes=((0,1,2,3),(3,1,2,0)))
    return Trace_AB


def initialize_A(T):
    # Make initial tensor of triangular lattice Ising model at a temperature T
    A =np.empty((2,2,2))

    for i in range(0,2):
        si = (i - 0.5) * 2
        for j in range(0,2):
            sj = (j - 0.5) * 2
            for k in range(0,2):
                sk = (k - 0.5) * 2
                if (i + j + k) % 2 == 0:
                    A[i,j,k] = 0.0
                else:
                    A[i,j,k] = np.exp((si + sj + sk)/(2.0*T))
                    
    ## normalization of tensor
    dt0 = np.ones(2,dtype=float)
    dt1 = np.ones(2,dtype=float)
    dt2 = np.ones(2,dtype=float)
    factor = Trace_tensor(A,A,dt0,dt1,dt2)

    factor_diag = (np.max(dt0)+np.max(dt1)+np.max(dt2)) / 3
    factor_tensor = np.sqrt(factor**(1.0/3.0)/factor_diag**3)

    
    A /= factor_tensor
    dt0 /= factor_diag
    dt1 /= factor_diag
    dt2 /= factor_diag
    
    return A,A.copy(),dt0,dt1,dt2,factor

def SVD_AB(A,B,dt,D,kp,bond=0):
    ## sigular value decomposition and truncation of A,B
    if bond ==1:
        A_mat = np.tensordot(np.tensordot(A,np.diag(dt),axes=(1,0)),B,axes=(2,1)).transpose(1,2,3,0).reshape(A.shape[0]*B.shape[2],B.shape[0]*A.shape[2])
    elif bond ==2:
        A_mat = np.tensordot(np.tensordot(A,np.diag(dt),axes=(2,0)),B,axes=(2,2)).transpose(0,3,2,1).reshape(A.shape[0]*B.shape[1],B.shape[0]*A.shape[1])
    else:
        A_mat = np.tensordot(np.tensordot(A,np.diag(dt),axes=(0,0)),B,axes=(2,0)).transpose(0,3,2,1).reshape(A.shape[1]*B.shape[2],B.shape[1]*A.shape[2])
        
    U,s,VT = linalg.svd(A_mat)

    ## truncate singular values at D_cut
    D_cut = np.min((s.size,D))
    s_t = s[0:D_cut]**kp

    if bond ==1:
        Sa = np.dot(U[:,0:D_cut],np.diag(s_t)).reshape(A.shape[0],B.shape[2],D_cut)
        Sb = np.dot(np.diag(s_t),VT[0:D_cut,:]).transpose(1,0).reshape(B.shape[0],A.shape[2],D_cut)
    elif bond ==2:
        Sa = np.dot(U[:,0:D_cut],np.diag(s_t)).reshape(A.shape[0], B.shape[1],D_cut)
        Sb = np.dot(np.diag(s_t),VT[0:D_cut,:]).transpose(1,0).reshape(B.shape[0],A.shape[1],D_cut)
    else:
        Sa = np.dot(U[:,0:D_cut],np.diag(s_t)).reshape(A.shape[1], B.shape[2],D_cut)
        Sb = np.dot(np.diag(s_t),VT[0:D_cut,:]).transpose(1,0).reshape(B.shape[1],A.shape[2],D_cut)
        
    return Sa,Sb,s[0:D_cut]/s_t**2



def SVD_AB_meanfield(A,B,dt,D):
    if dt is None:
        dt_sq = np.ones(A.shape[1])
    else:
        dt_sq = np.sqrt(dt)


        
    ## sigular value decomposition and truncation of A,B
    ## We assume symmetry
    A_mod = np.tensordot(np.tensordot(A,np.diag(dt_sq),axes=(1,0)),np.diag(dt_sq),axes=(1,0))
    B_mod = np.tensordot(np.tensordot(B,np.diag(dt_sq),axes=(1,0)),np.diag(dt_sq),axes=(1,0))

    A_mat = np.tensordot(A_mod,B_mod,axes=(0,0)).transpose(0,3,2,1).reshape(A.shape[1]*B.shape[2],B.shape[1]*A.shape[2])
        
    U,s,VT = linalg.svd(A_mat)

    ## truncate singular values at D_cut
    D_cut = np.min((s.size,D))
    #print("s_t = "+repr(s_t))

    if D_cut != s.size:
        D_cut_new = D_cut
        ##Test
        #diff = abs((s[D_cut_new-1]-s[D_cut_new])/s[D_cut_new])
        #while diff < 1e-3:
        #    D_cut_new -= 1
        #    diff = abs((s[D_cut_new-1]-s[D_cut_new])/s[D_cut_new])
        # 
        #truncation_error  = np.sum(s[D_cut:])/np.sqrt(np.sum(s**2))
        s_t = s[0:D_cut_new]        
        truncation_error  = np.sum(s[D_cut_new:])/np.sum(s)
    else:
        D_cut_new = D_cut
        s_t = s[0:D_cut_new]
        truncation_error = 0.0        
    #print("truncation error = "+repr(truncation_error))
    #print("s_t = "+repr(s_t)+", D_cut_new = "+repr(D_cut_new))

    dt_sq_inv = np.zeros(len(dt_sq))
    for i in range(len(dt_sq)):
        if dt_sq[i] > 0:
            dt_sq_inv[i] = 1.0/dt_sq[i]

    ##Test
    #for i in range(U.shape[0]):
    #    for j in range(D_cut_new):
    #        if abs(U[i,j]) < 1e-8:
    #            U[i,j] = 0.0
    #for i in range(D_cut_new):
    #    for j in range(VT.shape[1]):
    #        if abs(VT[i,j]) < 1e-8:
    #            VT[i,j] = 0.0                    
    


    
    Sa = np.tensordot(np.diag(dt_sq_inv),np.tensordot(np.diag(dt_sq_inv),np.dot(U[:,0:D_cut_new],np.diag(np.sqrt(s_t))).reshape(A.shape[1], B.shape[2],D_cut_new),axes=(1,1)),axes=(1,1))
    Sb = np.tensordot(np.diag(dt_sq_inv),np.tensordot(np.diag(dt_sq_inv),np.dot(np.diag(np.sqrt(s_t)),VT[0:D_cut_new,:]).transpose(1,0).reshape(B.shape[1],A.shape[2],D_cut_new),axes=(1,1)),axes=(1,1))

    ## because of the symmetry, the following gives the same results as the above, although Sa and Sb themselves are different from the original definitions.
    #Sa = np.tensordot(np.diag(dt_sq_inv),np.tensordot(np.diag(dt_sq_inv),np.dot(U[:,0:D_cut_new],np.diag(np.sqrt(s_t))).reshape(A.shape[1], B.shape[2],D_cut_new),axes=(1,0)),axes=(1,0))
    #Sb = np.tensordot(np.diag(dt_sq_inv),np.tensordot(np.diag(dt_sq_inv),np.dot(np.diag(np.sqrt(s_t)),VT[0:D_cut_new,:]).transpose(1,0).reshape(B.shape[1],A.shape[2],D_cut_new),axes=(1,0)),axes=(1,0))
    
    return Sa,Sb,s[:D_cut],truncation_error



def SVD_AB_meanfield_new(A,B,dt,D):

    if dt is None:
        ## SVD without env
        AB_mat = np.tensordot(A,B,axes=(0,0)).transpose(0,3,2,1).reshape(A.shape[1]*B.shape[2],B.shape[1]*A.shape[2])

        U,s,VT = linalg.svd(AB_mat)

        ## truncate singular values at D_cut
        D_cut = np.min((s.size,D))
        s_t = s[0:D_cut]
        print("s_t = "+repr(s_t))
        if D_cut != s.size:
            #truncation_error  = np.sum(s[D_cut:])/np.sqrt(np.sum(s**2))
            truncation_error  = np.sum(s[D_cut:])/np.sum(s)
        else:
            truncation_error = 0.0        
        print("truncation error = "+repr(truncation_error))

        Sa = np.dot(U[:,0:D_cut],np.diag(np.sqrt(s_t))).reshape(A.shape[1], B.shape[2],D_cut)
        Sb = np.dot(np.diag(np.sqrt(s_t)),VT[0:D_cut,:]).transpose(1,0).reshape(B.shape[1],A.shape[2],D_cut)

        return Sa,Sb,s_t,truncation_error

        
    else:
        dt_sq = np.sqrt(dt)
        ## sigular value decomposition and truncation of A,B
        ## We assume symmetry
        Me = np.kron(np.kron(np.kron(dt_sq,dt_sq),dt_sq),dt_sq).reshape(len(dt_sq)**2,len(dt_sq)**2)

        Ue,se,VeT = linalg.svd(Me)
        print("se = "+repr(se))

        Ue_s = np.dot(Ue,np.diag(np.sqrt(abs(se))))
        VeT_s = np.dot(np.diag(np.sqrt(abs(se))),VeT)

        AB_mat = np.dot(np.dot(VeT_s,np.tensordot(A,B,axes=(0,0)).transpose(0,3,2,1).reshape(A.shape[1]*B.shape[2],B.shape[1]*A.shape[2])),Ue_s)


        U,s,VT = linalg.svd(AB_mat)

        ## truncate singular values at D_cut
        D_cut = np.min((s.size,D))
        s_t = s[0:D_cut]

        print("s_t = "+repr(s_t))

        if D_cut != s.size:
            #truncation_error  = np.sum(s[D_cut:])/np.sqrt(np.sum(s**2))
            truncation_error  = np.sum(s[D_cut:])/np.sum(s)
        else:
            truncation_error = 0.0        
        print("truncation error = "+repr(truncation_error))


        Ue_inv = np.zeros(Ue.shape)
        VeT_inv = np.zeros(VeT.shape)
        for i in range(len(se)):
            if se[i]/se[0] > 1e-15:
                Ue_inv[:,i] = Ue.conj()[:,i]/np.sqrt(se[i])
                VeT_inv[i,:] = VeT.conj()[i,:]/np.sqrt(se[i])

        Sa = np.tensordot(VeT_inv,np.dot(U[:,0:D_cut],np.diag(np.sqrt(s_t))),axes=(0,0)).reshape(A.shape[1], B.shape[2],D_cut)
        Sb = np.tensordot(Ue_inv,np.dot(np.diag(np.sqrt(s_t)),VT[0:D_cut,:]),axes=(1,1)).reshape(B.shape[1],A.shape[2],D_cut)

        return Sa,Sb,s_t,truncation_error


def Calc_Entropy(A,B,dt0,dt1,dt2):
    dt0_sq = np.sqrt(dt0)
    dt1_sq = np.sqrt(dt1)
    dt2_sq = np.sqrt(dt2)

    A_mod = np.tensordot(np.tensordot(np.tensordot(A,np.diag(dt0_sq),axes=(0,0)),np.diag(dt1_sq),axes=(0,0)),np.diag(dt2_sq),axes=(0,0))
    B_mod = np.tensordot(np.tensordot(np.tensordot(B,np.diag(dt0_sq),axes=(0,0)),np.diag(dt1_sq),axes=(0,0)),np.diag(dt2_sq),axes=(0,0))

    entropy = np.zeros(3)
    for i in range(3):
        s_temp = linalg.svdvals(np.tensordot(A_mod,B_mod,axes=(i,i)).reshape(A.shape[(i+1)%3]*A.shape[(i+2)%3],B.shape[(i+1)%3]*A.shape[(i+2)%3]))
        s_temp /= np.sum(s_temp)
        log_s = np.zeros(len(s_temp))
        for j in range(len(s_temp)):
            if s_temp[j] > 0:
                log_s[j] = np.log(s_temp[j])
        entropy[i] = - np.dot(s_temp,log_s)
    
    return entropy
def Update_Atensor(A,B,dt0,dt1,dt2,D,kp,symmetric=True):    

    if symmetric:
        Sa,Sb,dt_new = SVD_AB(A,B,dt0,D,kp)
        A = Combine_three_S(Sa,Sa,Sa,dt0,dt1,dt2)
        B = Combine_three_S(Sb,Sb,Sb,dt0,dt1,dt2)
        dt0 = dt_new.copy()
        dt1 = dt_new.copy()
        dt2 = dt_new.copy()        
    else:
        Sa0,Sb0,dt0_new = SVD_AB(A,B,dt0,D,kp,bond=0)
        Sa1,Sb1,dt1_new = SVD_AB(A,B,dt1,D,kp,bond=1)
        Sa2,Sb2,dt2_new = SVD_AB(A,B,dt2,D,kp,bond=2)

        A = Combine_three_S(Sa0,Sa1,Sa2,dt0,dt1,dt2)
        B = Combine_three_S(Sb0,Sb1,Sb2,dt0,dt1,dt2)
        dt0 = dt0_new
        dt1 = dt1_new
        dt2 = dt2_new        


    factor = Trace_tensor(A,B,dt0,dt1,dt2)

    factor_diag = (np.max(dt0)+np.max(dt1)+np.max(dt2)) / 3
    factor_tensor = np.sqrt(factor**(1.0/3.0)/factor_diag**3)


    A /= factor_tensor
    B /= factor_tensor
    dt0 /= factor_diag
    dt1 /= factor_diag
    dt2 /= factor_diag

    return A, B, dt0,dt1,dt2,factor

def Update_Atensor_meanfield(A,B,dt,D,fixed_iteration,itr,epsilon):    

    if dt is None:
        dt_new = None
    else:
        dt_new = dt/dt[0]
    Sa,Sb,dt_new,truncation_error = SVD_AB_meanfield(A,B,dt_new,D)
    diff_truncation = 1.0
    
    D_edge = A.shape[1]
    if fixed_iteration:
        for i in range(itr):
            dt_new /= dt_new[0]
            Sa,Sb,dt_new,truncation_error = SVD_AB_meanfield(A,B,dt_new[:D_edge],D)
    else:
        while truncation_error > epsilon:# and diff_truncation > epsilon:
            dt_new /= dt_new[0]
            Sa,Sb,dt_new,truncation_error_new = SVD_AB_meanfield(A,B,dt_new[:D_edge],D)
            diff_truncation = abs((truncation_error_new - truncation_error)/truncation_error)
            truncation_error = truncation_error_new
            
    A = Combine_three_S_no_dt(Sa,Sa,Sa)
    B = Combine_three_S_no_dt(Sb,Sb,Sb)

    factor = Trace_tensor_no_dt(A,B)

    factor_tensor = np.sqrt(factor**(1.0/3.0))


    A /= factor_tensor
    B /= factor_tensor

    return A, B,dt_new,factor,truncation_error



def Combine_three_S(S0,S1,S2,dt0,dt1,dt2):
    S0_d = np.tensordot(np.diag(dt1),S0,axes=(0,0))
    S1_d = np.tensordot(np.diag(dt2),S1,axes=(0,0))
    S2_d = np.tensordot(np.diag(dt0),S2,axes=(0,0))
    
    A = np.tensordot(np.tensordot(S0_d,S1_d,axes=(1,0)),S2_d,axes=((2,0),(0,1)))
    return A

def Combine_three_S_no_dt(S0,S1,S2):
    A = np.tensordot(np.tensordot(S0,S1,axes=(1,0)),S2,axes=((2,0),(0,1)))
    return A

##### Main part of TRG ####
def TRG_Honeycomb_Ising(T,D,kp,TRG_steps,symmetric=True):

    ##  Initialization ##
    A,B,dt0,dt1,dt2,factor = initialize_A(T)

    entropy = Calc_Entropy(A,B,dt0,dt1,dt2)
    print("## Entroypy: "+repr(0)+" " +" ".join(map(str,entropy)))
    TRG_factors = [factor]
    
    ## TRG iteration ##
    for i_TRG in range(0,TRG_steps):

        A, B, dt0,dt1,dt2,factor = Update_Atensor(A,B,dt0,dt1,dt2,D,kp,symmetric)
        TRG_factors.append(factor)
        entropy = Calc_Entropy(A,B,dt0,dt1,dt2)
        print("## Entroypy: "+repr(i_TRG+1)+" " +" ".join(map(str,entropy)))

    ## End of TRG iteration

    #Calclation of free energy
    free_energy_density = 0.0
    for i_TRG in range(TRG_steps+1):
        free_energy_density +=  np.log(TRG_factors[i_TRG]) * (1.0/3.0)**i_TRG
    ## note: we normalize A so that Trace(A) = 1.0
    free_energy_density = -T * (1.0/3.0) * (free_energy_density + (1.0/3.0)**TRG_steps*np.log(Trace_tensor(A,B,dt0,dt1,dt2)))
    #Print("T, free_energy_density = "+repr(T)+" " +repr(free_energy_density))
    return free_energy_density

def SRG_Honeycomb_Ising(T,D,TRG_steps,fixed_iteration,itr,epsilon,use_previous_singular_value):

    ##  Initialization ##
    A,B,dt0,dt1,dt2,factor = initialize_A(T)

    TRG_factors = [factor]
    
    ## SRG iteration ##
    dt = None
    for i_TRG in range(0,TRG_steps):
        if use_previous_singular_value:
            A, B,dt,factor,truncation_error = Update_Atensor_meanfield(A,B,dt,D,fixed_iteration,itr,epsilon)
        else:
            A, B,dt,factor,truncation_error = Update_Atensor_meanfield(A,B,None,D,fixed_iteration,itr,epsilon)
        TRG_factors.append(factor)
    ## End of TRG iteration

    #Calclation of free energy
    free_energy_density = 0.0
    for i_TRG in range(TRG_steps+1):
        free_energy_density +=  np.log(TRG_factors[i_TRG]) * (1.0/3.0)**i_TRG
    ## note: we normalize A so that Trace(A) = 1.0
    free_energy_density = -T * (1.0/3.0) * (free_energy_density + (1.0/3.0)**TRG_steps*np.log(Trace_tensor_no_dt(A,B)))
    #Print("T, free_energy_density = "+repr(T)+" " +repr(free_energy_density))
    return free_energy_density
            
