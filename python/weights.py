import numpy as np

class Weights:
    def __init__( self, nodes ):
        self.nodes = nodes
        self.N  = len( nodes )
        self.A1 = np.array( np.zeros( ( self.N, self.N ) ) ) 

    def get_A1( self ):
        for i in range( 0, self.N ):
            for j in range( 0, self.N ):
                sum1 = 1
                sum2 = 1
                if j != i:
                    for k in range( 0, self.N ):
                        if i != k:
                            sum1 = sum1*( self.nodes[i] - self.nodes[k] )
                        if j != k:
                            sum2 = sum2*( self.nodes[j] - self.nodes[k] )
                    self.A1[i,j] = sum1/( sum2*( self.nodes[i] - self.nodes[j] ) )
                if j == self.N - 1:
                    self.A1[i,i] = -1*sum( self.A1[i,:] )
        return self.A1
                    
    def get_Higher( self, r, Ar ):
        A = np.array( np.zeros( ( self.N, self.N ) ) )
        for i in range( 0, self.N ):
            for j in range( 0, self.N ):
                if j != i:
                    A[i,j] = r*( Ar[i,i]*self.A1[i,j] - Ar[i,j]/( self.nodes[i] - self.nodes[j] ) )
                if j == self.N - 1:
                    A[i,i] = -1*sum( A[i,:] )
        return A
