# coding:utf-8

# Anisotropic Tensor Renormalization Groupe (ATRG) for square lattice Ising model
# based on D. Adachi, et al PRB (2020).
import numpy as np
import scipy as scipy
import scipy.linalg as linalg
import argparse
import ATRG

#input from command line
def parse_args():
    parser = argparse.ArgumentParser(description='Tensor Network Renormalization for Square lattice Ising model')
    parser.add_argument('-D', metavar='D',dest='D', type=int, default=4,
                        help='set bond dimension D for TRG')
    parser.add_argument('-n', metavar='n',dest='n', type=int, default=1,
                        help='set size n representing L=2^n')
    parser.add_argument('-T', metavar='T',dest='T', type=float, default=2.0,
                        help='set Temperature')
    #parser.add_argument('-k', metavar='kp',dest='kp', type=float, default=0.5,
    #                    help='set Temperature')
    #parser.add_argument('--HOTRG', action='store_const', const=True,
    #                    default=False, help='Parform HOTRG')
    parser.add_argument('--step', action='store_const', const=True,
                        default=False, help='Perform multi temperature calculation')
    parser.add_argument('--append', action='store_const', const=True,
                        default=False, help='Output is added to the exit file')
    parser.add_argument('-Tmin', metavar='Tmin',dest='Tmin', type=float, default=1.5,
                        help='set minimum temperature for step calculation')

    parser.add_argument('-Tmax', metavar='Tmax',dest='Tmax', type=float, default=3.0,
                        help='set maximum temperature for step calculation')
    parser.add_argument('-Tstep', metavar='Tstep',dest='Tstep', type=float, default=0.1,
                        help='set temperature increments for step calculation')    
    return parser.parse_args()

def Free_Energy_exact_2D_Ising(T):
    import scipy.integrate as integrate    
    def integrant(x,T):
        k = 1.0/np.sinh(2.0/T)**2
        k1 = 2.0*np.sqrt(k)/(1.0+k)
        result = np.log(2*(np.cosh(2.0/T)**2 + (k+1)/k*np.sqrt(1.0-k1**2*np.sin(x)**2)))
        return result

    k = 1.0/np.sinh(2.0/T)**2
    x,err =  integrate.quad(integrant, 0, np.pi*0.5, args=(T,),epsabs=1e-12,epsrel=1e-12)
    result = -T *x/np.pi

    return result,err * T/np.pi

def Calculate_TRG(D=4,n=1,T=2.0,Tmin=1.5,Tmax=3.0,Tstep=0.1,Step_flag=False,Append_flag=False):

    TRG_step = 2*n - 1
    L = 2**n
    
    tag = "_L"+str(L)+"_D"+str(D)+"_ATRG"
    
    if Append_flag:
        file_free_energy = open("free_energy"+tag+".dat","a")
    else:
        file_free_energy = open("free_energy"+tag+".dat","w")

    file_free_energy.write("# L = "+repr(L) + ", TRG_step = "+repr(TRG_step)+ "\n")
    print("# L = "+repr(L) + ", TRG_step = "+repr(TRG_step))

    if Step_flag:
        ## step calculation
        for T in np.arange(Tmin,Tmax,Tstep):
            f_ex, err = Free_Energy_exact_2D_Ising(T)
            free_energy_density = ATRG.ATRG_Square_Ising(T,D,TRG_step)

            print(repr(T)+" "+repr(D)+" "+repr(free_energy_density) + " " + repr(f_ex) + " " + repr(err)+" "+repr(free_energy_density-f_ex))
            file_free_energy.write(repr(T)+" "+repr(D)+" "+repr(free_energy_density) + " " + repr(f_ex) + " " + repr(err)+" "+repr(free_energy_density-f_ex)+"\n")                    
    else:
        ## single calculation
        f_ex, err = Free_Energy_exact_2D_Ising(T)
        free_energy_density = ATRG.ATRG_Square_Ising(T,D,TRG_step)
        print(repr(T)+" "+repr(D)+" "+repr(free_energy_density) + " " + repr(f_ex) + " " + repr(err)+" "+repr(free_energy_density-f_ex))
        file_free_energy.write(repr(T)+" "+repr(D)+" "+repr(free_energy_density) + " " + repr(f_ex) + " " + repr(err)+" "+repr(free_energy_density-f_ex)+"\n")                    

def main():
    ## read params from command line
    args = parse_args()
    Calculate_TRG(args.D,args.n,args.T,args.Tmin,args.Tmax,args.Tstep,args.step,args.append)    
    
if __name__ == "__main__":
    main()
