# coding:utf-8

# Tensor Renormalization Groupe (TRG) and Bond-weighted TRG (BTRG) for honeycomb lattice Ising

import numpy as np
import scipy as scipy
import scipy.linalg as linalg
import argparse
import TRG_honeycomb

#input from command line
def parse_args():
    Tc = 4.0/np.log(3.0)
    parser = argparse.ArgumentParser(description='Tensor Network Renormalization for triangular lattice Ising model')
    parser.add_argument('-D', metavar='D',dest='D', type=int, default=4,
                        help='set bond dimension D for TRG, (default: D = 4)')
    parser.add_argument('-n', metavar='n',dest='n', type=int, default=5,
                        help='set size n representing N=3^{n+1}, (default: n = 5)')
    parser.add_argument('-T', metavar='T',dest='T', type=float, default=Tc,
                        help='set Temperature (default: Tc=4/log(3))')
    parser.add_argument('-k', metavar='kp',dest='kp', type=float, default=0.5,
                        help='set Temperature')
    parser.add_argument('--step', action='store_const', const=True,
                        default=False, help='Perform multi temperature calculation, (default: False)')
    parser.add_argument('--append', action='store_const', const=True,
                        default=False, help='Output is added to the exit file, (default: False)')
    parser.add_argument('--no_symmetric', action='store_const', const=True,
                        default=False, help='TRG without assuming lattice rotational symmetry, (default: False (symmetric)')
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

    x,err =  integrate.dblquad(integrant, 0, np.pi*2, lambda x: 0,lambda x: np.pi*2, args=(T,),epsabs=1e-12,epsrel=1e-12)

    result = -T  * ( 1.0/T +np.log(2*np.cosh(2.0/T)) +x/(8*np.pi**2))

    return result,err * T/(8*np.pi**2)

def Calculate_TRG(D=4,n=1,kp=0.5,T=2.0,Tmin=1.5,Tmax=3.0,Tstep=0.1,symmetric=True,Step_flag=False,Append_flag=False):

    TRG_step = n
    N = 3**(n+1)
    
    tag = "_n"+str(n)+"_D"+str(D)
    
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
            free_energy_density = TRG_honeycomb.TRG_Honeycomb_Ising(T,D,kp,TRG_step,symmetric)

            print(repr(T)+" "+repr(D)+" "+repr(kp)+" "+repr(free_energy_density) + " " + repr(f_ex) + " " + repr(err)+" "+repr(free_energy_density-f_ex))
            file_free_energy.write(repr(T)+" "+repr(D)+" "+repr(kp)+" "+repr(free_energy_density) + " " + repr(f_ex) + " " + repr(err)+" "+repr(free_energy_density-f_ex)+"\n")                    
    else:
        ## single calculation
        f_ex, err = Free_Energy_exact_triangular_Ising(T)
        free_energy_density = TRG_honeycomb.TRG_Honeycomb_Ising(T,D,kp,TRG_step,symmetric)
        print(repr(T)+" "+repr(D)+" "+repr(kp)+" "+repr(free_energy_density) + " " + repr(f_ex) + " " + repr(err)+" "+repr(free_energy_density-f_ex))
        file_free_energy.write(repr(T)+" "+repr(D)+" "+repr(kp)+" "+repr(free_energy_density) + " " + repr(f_ex) + " " + repr(err)+" "+repr(free_energy_density-f_ex)+"\n")                    

def main():
    ## read params from command line
    args = parse_args()
    Calculate_TRG(args.D,args.n,args.kp,args.T,args.Tmin,args.Tmax,args.Tstep,not args.no_symmetric,args.step,args.append)    
    
if __name__ == "__main__":
    main()
