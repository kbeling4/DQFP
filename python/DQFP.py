# Straight Ahead Fokker-Plank Model using the first two moments
# Units:    length [=] cm
#           energy [=] MeV

import particle as prt
import material as mat
import grid as gd
import weights as wt
import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg


def main():
    # ----- Setup Problem ----------------------------------
    particle = prt.Particle()
    material = mat.Material()

    Emin =     100
    Emax =     120
    Ne   =       9

    Zmin =      0
    Zmax =   1000
    Nz   =     25
    
    # ----- Calculate matrix A ------------------------------------
    grid = gd.Grid( Ne, Emin, Emax, Nz, Zmin, Zmax )

    Ewts = wt.Weights( grid.Enodes )
    B1 = Ewts.get_A1()
    B2 = Ewts.get_Higher( 2, B1 )
    grid.get_Agrid( B1, B2, particle, material, 0 )
    
    # ----- Initial Spectrum -------------------------------
    Yi    = np.zeros( Ne ) - 10
    # print( grid.Agrid )
    
    # ----- Solver ---------------------------------------------
    B = np.array( Yi )

    for i in range( 0, 10 ):
        x = linalg.solve( grid.Agrid, B )
        grid.phi[:,i] = x
        B = (-1)*x
        # if z % 100 == 0:
        #     print( 'z iteration: ', z )

    plt.plot( grid.Enodes, grid.phi[:,1] )
    plt.plot( grid.Enodes, grid.phi[:,2] )
    plt.plot( grid.Enodes, grid.phi[:,3] )
    plt.show()
        
if __name__ == "__main__": main()
