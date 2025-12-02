# Installation d'abord : pip install dtw-python
"""
from dtw import dtw
import numpy as np

# Avec la bibliothèque dtw-python
def dtw_library(Q, C):
    Q = np.array(Q).reshape(-1, 1)
    C = np.array(C).reshape(-1, 1)
    
    # Calcul DTW
    alignment = dtw(Q, C, keep_internals=True)
    
    print(f"Distance DTW: {alignment.distance}")
    print(f"Chemin optimal: {alignment.index1} ↔ {alignment.index2}")
    
    # Visualisation
    alignment.plot(type="twoway")
    
    return alignment.distance

# Exemple plus complexe
Q_long = [1, 3, 4, 9, 10, 3, 2]
C_long = [1, 2, 3, 4, 5, 6, 7, 8]

print("Exemple avec séries de longueurs différentes:")
print(f"Longueur Q: {len(Q_long)}, Longueur C: {len(C_long)}")

# Avec notre fonction basique
distance = dtw_fast(Q_long, C_long)
print(f"Distance DTW: {distance}")

"""

import numpy as np

def dtw_basic(Q, C):
    """
    Implémentation basique de DTW
    Q, C: listes ou tableaux numpy
    """
    m, n = len(Q), len(C)
    
    # Créer la matrice D
    D = np.zeros((m+1, n+1))
    
    # Initialisation
    D[0, 0] = 0
    for i in range(1, m+1):
        D[i, 0] = float('inf')
    for j in range(1, n+1):
        D[0, j] = float('inf')
    
    # Remplissage
    for i in range(1, m+1):
        for j in range(1, n+1):
            cost = abs(Q[i-1] - C[j-1])
            D[i, j] = cost + min(D[i-1, j],    # insertion
                                 D[i, j-1],    # suppression
                                 D[i-1, j-1])  # match
    
    return D[m, n], D

# Test avec notre exemple
Q = [1, 3, 4]
C = [1, 2, 3]

distance, matrice = dtw_basic(Q, C)
print(f"Distance DTW: {distance}")
print("Matrice D:")
print(matrice)
