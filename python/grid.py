import numpy as np

class Grid:
    def __init__(self, Ne, Emin, Emax, Nz, Zmin, Zmax):
        self.Ne     = Ne
        self.Nz     = Nz
        self.Emin   = Emin
        self.Emax   = Emax
        self.Zmin   = Zmin
        self.Zmax   = Zmax
        self.Enodes = np.zeros( Ne )
        self.Znodes = np.linspace( Zmin, Zmax, self.Nz, endpoint=True)
        self.Sgrid  = np.array( np.zeros( ( self.Ne, self.Ne ) ) )
        self.Tgrid  = np.array( np.zeros( ( self.Ne, self.Ne ) ) )
        self.Agrid  = np.array( np.zeros( ( self.Ne, self.Ne ) ) )
        self.phi    = np.array( np.zeros( ( self.Ne, self.Nz ) ) )

    def get_Enodes( self ):
        self.Enodes[0]         = self.Emin
        self.Enodes[self.Ne-1] = self.Emax
        for i in range( 2, self.Ne ):
            self.Enodes[i-1] = self.Emin + 0.5*( 1 - np.cos( (i - 1)/(self.Ne - 1)*np.pi ) )*(self.Emax - self.Emin)
        return self.Enodes

    def find_Enode( self, value):
        return (np.abs(self.Enodes - value)).argmin()

    def find_Znode( self, value):
        return (np.abs(self.Znodes - value)).argmin()
    
    def S( self, E, particle, material ):
        particle.get_qmax( E )
        amp = 0.1536*(particle.Z)**2*material.Z*material.rho/(material.M*particle.beta2)
        return amp*( np.log( particle.qmax/material.qmin ) - particle.beta2*( 1 - material.qmin/particle.qmax ) )

    def T( self, E, particle, material ):
        particle.get_qmax( E )
        amp  = 0.1536*(particle.Z)**2*material.Z*material.rho/(material.M*particle.beta2)
        return amp*( particle.qmax*( 1-particle.beta2/2 ) - material.qmin*( 1- (particle.beta2/2)
                                                                            *(material.qmin/particle.qmax) ) )

    def get_Sgrid( self, particle, material ):
        for i in range(0, self.Ne ):
            for j in range(0, self.Ne ):
                    self.Sgrid[i,j] = self.S( self.Enodes[j], particle, material )

    def get_Tgrid( self, particle, material ):
        for i in range(0, self.Ne ):
            for j in range(0, self.Ne ):
                if j+1 == i:
                    self.Tgrid[i,j] = self.T( self.Enodes[i-1], particle, material )
                if i == j:
                    self.Tgrid[i,j] = self.T( self.Enodes[i], particle, material )
                if i+1 == j:
                    self.Tgrid[i,j] = self.T( self.Enodes[i+1], particle, material )

    def get_Agrid( self, A1, A2, particle, material, z ):
        for i in range(0, self.Ne ):
            for j in range(0, self.Ne ):
                del_z = self.Znodes[z+1] - self.Znodes[z]
                if j != i:
                    S = self.S( self.Enodes[j], particle, material )
                    T = self.T( self.Enodes[j], particle, material )
                    self.Agrid[i,j] = del_z*(A1[i,j]*S + 0.5*A2[i,j]*T)
                    
                if i == j:
                    S = self.S( self.Enodes[j], particle, material )
                    T = self.T( self.Enodes[j], particle, material )
                    self.Agrid[i,i] = del_z*(A1[i,j]*S + 0.5*A2[i,j]*T) - 1
