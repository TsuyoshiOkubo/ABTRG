# coding:utf-8

# Tensor Renormalization Groupe (TRG) with mena-field SRG for square lattice Ising

import numpy as np
import scipy as scipy
import scipy.linalg as linalg
import argparse
import TRG_honeycomb

#input from command line
Tc = 4.0/np.log(3.0)
def parse_args():
    Tc = 4.0/np.log(3.0)
    parser = argparse.ArgumentParser(description='Tensor Network Renormalization for triangular lattice Ising model')
    parser.add_argument('-D', metavar='D',dest='D', type=int, default=4,
                        help='set bond dimension D for TRG, (default: D = 4)')
    parser.add_argument('-n', metavar='n',dest='n', type=int, default=5,
                        help='set size n representing N=3^{n+1}, (default: n = 5)')
    parser.add_argument('-T', metavar='T',dest='T', type=float, default=Tc,
                        help='set Temperature (default: Tc=4/log(3))')
    parser.add_argument('-itr', metavar='itr',dest='itr', type=int, default=0,
                        help='set itration for meanfield SRG. It is used only if fixed iteration is true, (default: itr = 0, corresponding TRG)')
    parser.add_argument('--fixed_iteration', action='store_const', const=True,
                        default=False, help='Use fixed itration number for the iteration of SRG mean field, (default: False)')
    parser.add_argument('-epsilon', metavar='epsilon',dest='epsilon', type=float, default=1e-8,
                        help='set epsilon to determin the iteration of  meanfield SRG, (default: epsilon = 1e-8)')
    parser.add_argument('--prev_svd', action='store_const', const=True,
                        default=False, help='Use previous singular values for the iteration of SRG mean field, (default: False)')
    parser.add_argument('--step', action='store_const', const=True,
                        default=False, help='Perform multi temperature calculation, (default: False)')
    parser.add_argument('--append', action='store_const', const=True,
                        default=False, help='Output is added to the exit file, (default: False)')
    parser.add_argument('-Tmin', metavar='Tmin',dest='Tmin', type=float, default=2.5,
                        help='set minimum temperature for step calculation, (default: Tmin = 2.5)')

    parser.add_argument('-Tmax', metavar='Tmax',dest='Tmax', type=float, default=4.5,
                        help='set maximum temperature for step calculation, (default: Tmax = 4.5)')
    parser.add_argument('-Tstep', metavar='Tstep',dest='Tstep', type=float, default=0.1,
                        help='set temperature increments for step calculation, (default: Tstep = 0.1)')    
    return parser.parse_args()

def Free_Energy_exact_triangular_Ising(T):
    ## from Wannier PR, 79, 357 (1950), Eq.32
    import scipy.integrate as integrate    

    def integrant(y,x,T):
        k = 0.5*np.tanh(2.0/T)/np.cosh(2.0/T) * np.exp(-2.0/T)        
        #k = (np.exp(4.0/T)-1)/(np.exp(4.0/T)+1)**2
        result = np.log(1.0 + 4*k * np.cos(y) * (np.cos(x) - np.cos(y)))
        return result

    x,err =  integrate.dblquad(integrant, 0, np.pi*2, lambda x: 0,lambda x: np.pi*2,args=(T,),epsabs=1e-12,epsrel=1e-12)

    result = -T  * ( 1.0/T +np.log(2*np.cosh(2.0/T)) +x/(8*np.pi**2))

    return result,err * T/(8*np.pi**2)

def Calculate_SRG(D=4,n=5,fixed_iteration = True, itr=0, epsilon = 1e-8,prev_svd = False,T=Tc,Tmin=2.5,Tmax=4.5,Tstep=0.1,Step_flag=False,Append_flag=False):

    TRG_step = n
    N = 3**(n+1)
    
    tag = "_n"+str(n)+"_D"+str(D)+"_SRG"
    
    if Append_flag:
        file_free_energy = open("free_energy"+tag+".dat","a")
    else:
        file_free_energy = open("free_energy"+tag+".dat","w")

    file_free_energy.write("# TRG_step = "+repr(TRG_step)+ "\n")
    print("# TRG_step = "+repr(TRG_step))

    if Step_flag:
        ## step calculation
        for T in np.arange(Tmin,Tmax,Tstep):
            f_ex, err = Free_Energy_exact_triangular_Ising(T)
            free_energy_density = TRG_honeycomb.SRG_Honeycomb_Ising(T,D,TRG_step,fixed_iteration,itr,epsilon,prev_svd)

            print(repr(T)+" "+repr(D)+" "+repr(itr)+" "+repr(free_energy_density) + " " + repr(f_ex) + " " + repr(err)+" "+repr(free_energy_density-f_ex))
            file_free_energy.write(repr(T)+" "+repr(D)+" "+repr(itr)+" "+repr(free_energy_density) + " " + repr(f_ex) + " " + repr(err)+" "+repr(free_energy_density-f_ex)+"\n")                    
    else:
        ## single calculation
        f_ex, err = Free_Energy_exact_triangular_Ising(T)
        free_energy_density = TRG_honeycomb.SRG_Honeycomb_Ising(T,D,TRG_step,fixed_iteration,itr,epsilon,prev_svd)
        print(repr(T)+" "+repr(D)+" "+repr(itr)+" "+repr(free_energy_density) + " " + repr(f_ex) + " " + repr(err)+" "+repr(free_energy_density-f_ex))
        file_free_energy.write(repr(T)+" "+repr(D)+" "+repr(itr)+" "+repr(free_energy_density) + " " + repr(f_ex) + " " + repr(err)+" "+repr(free_energy_density-f_ex)+"\n")                    

def main():
    ## read params from command line
    args = parse_args()
    Calculate_SRG(args.D,args.n,args.fixed_iteration,args.itr,args.epsilon,args.prev_svd,args.T,args.Tmin,args.Tmax,args.Tstep,args.step,args.append)    
    
if __name__ == "__main__":
    main()
