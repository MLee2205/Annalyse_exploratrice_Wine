import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import SpectralCoclustering

def perform_biclustering_example():
    """
    Crée une matrice de données simple, applique le biclustering, 
    et affiche les matrices originale et réordonnée.
    """
    print("=== Démarrage du Biclustering Spectral ===")
    
    # --- 1. Création de la matrice de données d'exemple ---
    # Cette matrice est conçue pour avoir 2x2 bi-clusters clairs:
    # Bloc 1 (en haut à gauche) : Valeurs faibles (autour de 1)
    # Bloc 2 (en bas à droite) : Valeurs moyennes (autour de 5)
    # Blocs 3 et 4 : Valeurs élevées (autour de 9)
    
    X = np.array([
        # Lignes 0, 1 (faibles & élevées)
        [1, 2, 1, 9, 8],
        [2, 1, 2, 9, 9],
        # Lignes 2, 3 (élevées & moyennes)
        [8, 9, 9, 5, 6],
        [9, 8, 8, 6, 5]
    ])
    
    n_rows, n_cols = X.shape
    print(f"Matrice de données (Lignes: {n_rows}, Colonnes: {n_cols}) :\n{X}")

    # --- 2. Application du Biclustering ---
    # Nous cherchons 2 clusters de lignes et 2 clusters de colonnes (2x2 = 4 bi-clusters)
    n_biclusters = 2 
    model = SpectralCoclustering(n_clusters=n_biclusters, random_state=42)
    model.fit(X)

    # --- 3. Affichage des Labels (Étiquettes) ---
    print("\n--- Résultats des Labels ---")
    print(f"Labels des Lignes : {model.row_labels_}")
    print(f"Labels des Colonnes : {model.column_labels_}")

    # --- 4. Réordonnancement et Visualisation ---
    # Pour visualiser les bi-clusters, nous devons réorganiser la matrice
    
    # a) Trier les lignes selon leurs labels
    row_sort_indices = np.argsort(model.row_labels_)
    fit_data = X[row_sort_indices]

    # b) Trier les colonnes selon leurs labels
    column_sort_indices = np.argsort(model.column_labels_)
    fit_data = fit_data[:, column_sort_indices]

    # Affichage des matrices
    plt.figure(figsize=(10, 4))
    
    # 4.1 Matrice Originale
    plt.subplot(1, 2, 1)
    plt.imshow(X, cmap='viridis', aspect='auto')
    plt.title("Matrice Originale")
    plt.xlabel('Colonnes')
    plt.ylabel('Lignes')

    # 4.2 Matrice Réordonnée (Bi-clusters visibles)
    plt.subplot(1, 2, 2)
    plt.imshow(fit_data, cmap='viridis', aspect='auto')
    plt.title("Matrice Réordonnée par Biclustering")
    plt.xlabel('Colonnes (triées)')
    plt.ylabel('Lignes (triées)')

    # Ajout de lignes rouges pour délimiter les bi-clusters (pour interprétation)
    # La ligne se place après la première moitié des lignes/colonnes triées
    row_midpoint = n_rows / n_biclusters - 0.5
    col_midpoint = n_cols * model.column_labels_.tolist().count(0) / n_cols - 0.5
    
    plt.axhline(y=row_midpoint, color='red', linestyle='--', linewidth=2)
    
    # Déterminer la position de la coupure de colonne
    # Les labels des colonnes sont [0, 0, 0, 1, 1], la coupure est après l'index 2.
    # Dans la matrice réordonnée, la coupure est entre les colonnes triées.
    num_cols_cluster0 = np.sum(model.column_labels_ == 0)
    plt.axvline(x=num_cols_cluster0 - 0.5, color='red', linestyle='--', linewidth=2)
    
    plt.tight_layout()
    plt.show()
    
    print("\nRegardez le graphique : la matrice réordonnée (à droite) montre 4 blocs colorés distincts.")
    
    
# Exécuter l'exemple
if __name__ == "__main__":
    perform_biclustering_example()
    
    
"""

Explication du Résultat

Le biclustering ne produit pas de cluster unique de points, mais un ensemble d'étiquettes de lignes et de colonnes qui, lorsqu'elles sont combinées, définissent des régions rectangulaires (les bi-clusters) dans la matrice de données.

    Labels des Lignes : Indiquent quels échantillons (lignes) sont similaires entre eux.

        Exemple : [1, 1, 0, 0] signifie que les lignes 0 et 1 forment un groupe, et les lignes 2 et 3 forment un autre groupe.

    Labels des Colonnes : Indiquent quelles caractéristiques (colonnes) sont similaires entre elles.

        Exemple : [0, 0, 0, 1, 1] signifie que les colonnes 0, 1, et 2 sont similaires, et les colonnes 3 et 4 sont similaires.

    Visualisation : La seule façon de voir les bi-clusters est de réordonner la matrice en triant d'abord les lignes selon model.row_labels_, puis les colonnes selon model.column_labels_. Cela fait apparaître les quatre blocs distincts (les bi-clusters) dans la matrice réordonnée.
    
"""
