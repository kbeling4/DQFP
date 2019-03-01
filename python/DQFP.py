# Straight Ahead Fokker-Plank Model using the first two moments
# Units:    length [=] cm
#           energy [=] MeV

import particle as prt
import material as mat
import grid as gd
import weights as wt
import weights2 as wt2
import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg


def main():
    # ----- Setup Problem ----------------------------------
    particle = prt.Particle()
    material = mat.Material()

    Emin =   1550
    Emax =   1800
    Ne   =   62

    Zmin =      0
    Zmax =     25
    Nz   =    100

    # ----- Calculate matrix A ------------------------------------
    grid = gd.Grid( Ne, Emin, Emax, Nz, Zmin, Zmax )
    Enodes = grid.get_Enodes()
    Ewts2 = wt2.Weights2( Enodes )
    A1 = Ewts2.get_A1()
    A2 = Ewts2.get_A2()

    grid.get_Agrid( A1, A2, particle, material, 0 )
    
    # ----- Initial Spectrum -------------------------------
    Start = 1705
    idx = grid.find_Enode( Start )
    Yi = np.zeros( Ne )
    Yi[idx] = -1
    B = np.array( Yi )
    # print( grid.Enodes[idx] )
    
    # ----- Solver ---------------------------------------------
    for i in range( 0, Nz ):
        x = linalg.solve( grid.Agrid, B )
        grid.phi[:,i] = x
        B = (-1)*x
        if i % 100 == 0:
            print( 'step: ', i )

    idx4 = grid.find_Znode( 5.0 )
    # print( grid.Znodes[idx4] )
    # np.savetxt('output.txt', np.column_stack((grid.Enodes, grid.phi[:,idx4]) ), fmt="%1.4e", delimiter=' ')

    plt.figure( 1 )
    plt.plot( grid.Enodes, grid.phi[:,idx4], 'g' )
    plt.plot( grid.Enodes, grid.phi[:,idx4], 'o' )

    plt.show()

        
if __name__ == "__main__": main()
