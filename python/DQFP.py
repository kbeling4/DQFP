# Straight Ahead Fokker-Plank Model using the first two moments
# Units:    length [=] cm
#           energy [=] MeV

import particle as prt
import material as mat
import grid as gd
import weights2 as wt2
import spectrum as spec

import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg

def main():
    # ----- Setup Problem ----------------------------------
    particle = prt.Particle()
    material = mat.Material()

    Emin =   100
    Emax =   1000
    Ne   =   100

    Zmin =      0
    Zmax =     30
    Nz   =    1000

    # ----- Calculate matrix A ------------------------------------
    grid = gd.Grid( Ne, Emin, Emax, Nz, Zmin, Zmax )
    Enodes = grid.get_EnodesChebyshev()
    Ewts2 = wt2.Weights2( Enodes )
    A1 = Ewts2.get_A1()
    A2 = Ewts2.get_A2()
    grid.get_Agrid( A1, A2, particle, material, 0 )
    
    # ----- Initial Spectrum -------------------------------
    idx1 = grid.find_Enode( Emax )
    idx2 = grid.find_Enode( Emin )

    Spec = spec.Spectrum( Enodes, idx1, idx2 )
    # Spec.normalizer()
    #grid.get_b( Spec.gaussian( 700, 100 ) )

    B = -1*np.array( Spec.gaussian( 700, 500 ) )

    B1 = -1*B
    #B = -1*grid.b
    # ----- Solver ---------------------------------------------
    for i in range( 0, Nz ):
        x = linalg.solve( grid.Agrid, B )
        grid.phi[:,i] = x
        B = (-1)*x
        if i % 100 == 0:
            print( 'step: ', i )

    idx4 = grid.find_Znode( 25.0 )
    np.savetxt('output.txt', np.column_stack((grid.Enodes, grid.phi[:,idx4]) ), fmt="%1.4e", delimiter=' ')
    #np.savetxt('output.txt', np.column_stack((grid.Enodes, B1) ), fmt="%1.4e", delimiter=' ')

    plt.figure( 1 )
    plt.plot( grid.Enodes, B1, 'r' )
    plt.plot( grid.Enodes, grid.phi[:,idx4], 'g' )
    plt.plot( grid.Enodes, grid.phi[:,idx4], 'o' )

    plt.show()

        
if __name__ == "__main__": main()
