import numpy as np

def ThomasAlg_Vek( A, B, C, D ):
    """
    Thomas-Algorithmus, siehe https://de.wikipedia.org/wiki/Thomas-Algorithmus
    Vektoren sind
    A = untere Nebendiagonale,  B = Hauptdiagonale 
    C = obere Nebendiagonale, D = rechte Seite 
    X = ErgebnisVektor
    ACHTUNG: 
    Dimension der Hauptdiagonale B = n  
    Dimension der Nebendiagonalen A, C = n-1 !!
    --> statt https://de.wikipedia.org/wiki/Thomas-Algorithmus: A(i-1) in Formeln
    Dimension der rechten Seite D = n
    Dimension der Loesung X = n
    """
    
    # Dimension des GLS    
    n = len(D)
    # Speicher fuer Loesung
    X = np.zeros(n, dtype=float)
    # Speicher für Neue modifizierte Vektoren im Vorwaerts-Durchlauf
    Cneu = np.zeros(n-1, dtype=float)
    Dneu = np.zeros(n,   dtype=float)
    
    # Gauss-Elimination Vorwaerts-Durchlauf
    # Gleichung 1 als Index 0 in Python
    Cneu[0] = C[0] / B[0]
    Dneu[0] = D[0] / B[0]
    
    # Gleichungen 2 ... n-1 als Indizes 1 ... n-2 in Python
    for i in range (1, n-1, 1):
        temp = B[i] - (Cneu[i-1] * A[i-1])        
        Cneu[i] = C[i] / temp
        Dneu[i] = ( D[i] - Dneu[i-1] * A[i-1] ) / temp
    
    # Gleichung n als Index n-1 (bzw. -1) in Python
    temp = B[-1] - (Cneu[n-2] * A[n-2])  
    Dneu[-1] = ( D[-1] - Dneu[n-2] * A[n-2] ) / temp
    
            
    # Rueckwarts-Durchlauf
    # Gleichung n als Index n-1 (bzw. -1) in Python
    X[-1] = Dneu[-1]
    # Gleichungen n-1 ... 1 als Indizes n-2 .. 0  in Python
    for i in range(n-2, -1, -1):
        X[i] = Dneu[i] - Cneu[i] * X[i+1]
    
    return(X)

def ThomasAlg_Mat( A, d ):
    """
    Thomas-Algorithmus, siehe https://de.wikipedia.org/wiki/Thomas-Algorithmus
    Matrix und Vektoren fuer Tridiagonal-GLS: A * x = d
    A = quadratische Tridiagonal-Matrix 
    d = rechte Seite (Vektor) 
    x = Ergebnis-Vektor    
    """
    
    # Dimension des GLS    
    n = len(d)
    # Speicher fuer Loesung
    X = np.zeros(n, dtype=float)
    # Speicher für Neue modifizierte Vektoren im Vorwaerts-Durchlauf
    Cneu = np.zeros(n, dtype=float)
    Dneu = np.zeros(n,   dtype=float)
    
    # Gauss-Elimination Vorwaerts-Durchlauf
    # Gleichung 1 als Index 0 in Python
    Cneu[0] = A[0,1] / A[0,0]
    Dneu[0] = d[0] / A[0,0]
    
    # Gleichungen 2 ... n-1 als Indizes 1 ... n-2 in Python
    for i in range (1, n-1, 1):
        temp = A[i,i] - (Cneu[i-1] * A[i,i-1])        
        Cneu[i] = A[i,i+1] / temp
        Dneu[i] = ( d[i] - Dneu[i-1] * A[i,i-1] ) / temp
    
    # Gleichung n als Index n-1 (bzw. -1) in Python
    temp = A[-1,-1] - (Cneu[n-2] * A[n-1,n-2])  
    Dneu[-1] = ( d[-1] - Dneu[n-2] * A[n-1,n-2] ) / temp
                
    # Rueckwarts-Durchlauf
    # Gleichung n als Index n-1 (bzw. -1) in Python
    X[-1] = Dneu[-1]
    # Gleichungen n-1 ... 1 als Indizes n-2 .. 0  in Python
    for i in range(n-2, -1, -1):
        X[i] = Dneu[i] - Cneu[i] * X[i+1]
    
    return(X)

#### Testbeispiel
if __name__ == '__main__':
    
    import math
    n=4
    
    # A as matrix of dimension n=4
    A = np.array([[2,3,0,0],[4,5,6,0],[0,7,8,9],[0,0,10,11]],dtype=float)   
    
    a = np.array([   4, 7, 10], dtype=float) #  subdiagonal
    b = np.array([2, 5, 8, 11], dtype=float) # main diagonal
    c = np.array([3, 6, 9    ], dtype=float) #  super diagonal
    
    d = np.array([8, 32, 74, 74 ], dtype=float) # Rechte Seite des GLS

    # Thomas Alg in Vektor-Form
    X_vec = ThomasAlg_Vek(a, b, c, d)
    print(X_vec)
    
    # Thomas Alg in Matrix-Form
    X_mat = ThomasAlg_Mat(A, d)
    print(X_mat)
    
    # Thomas Alg in Matrix-Form, sparse tridiagonal Matrix
    from scipy import sparse # basis of sparse matrix
    diags = [-1, 0, 1]; #  first subdiagonal, main diagonal, first super diagonal

    # Diagonals as rows, using spdiags
    Diagonals_row = np.vstack([ np.insert(a, len(a), math.nan), b, np.insert(c, 0, math.nan)])
    # Note : Rows instead of Columns (Matlab), format = csc or csr for elementwise access
    A_spa = sparse.spdiags(Diagonals_row, diags, n,n, format="csc") 
    print(A_spa.todense() )
    
    # Diagonals as sequence of np.arrays, using diags    
    A_spa = sparse.diags([ a, b, c], diags, shape=(n,n), format="csc") 
    X_spa = ThomasAlg_Mat(A_spa, d)
    print(X_spa)

    # Comparison with numpy linear algebra library
    print(np.linalg.solve(A, d) )


