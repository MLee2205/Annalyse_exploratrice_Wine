import numpy as np
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.neighbors import KDTree
from sklearn.datasets import make_blobs, make_moons, make_circles
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
import warnings

# --- Classe DENCLUE Optimisée ---

class DenclueOptimized(BaseEstimator, ClusterMixin):
    """
    Version optimisée de DENCLUE utilisant des grilles et KD-Trees.
    """
    
    def __init__(self, sigma=0.5, xi=0.1, n_grid=50, min_density=5, 
                 step_size=0.1, max_iter=100):
        self.sigma = sigma
        self.xi = xi
        self.n_grid = n_grid
        self.min_density = min_density
        self.step_size = step_size
        self.max_iter = max_iter
        self.labels_ = None
        self.attractors_ = None
        
    def _create_grid(self, X):
        """Crée une grille pour accélérer les calculs."""
        mins = X.min(axis=0)
        maxs = X.max(axis=0)
        
        # Créer la grille
        grid_axes = [np.linspace(mins[i], maxs[i], self.n_grid) 
                      for i in range(X.shape[1])]
        grid = np.meshgrid(*grid_axes)
        grid_points = np.vstack([g.ravel() for g in grid]).T
        
        return grid_points
    
    def _compute_grid_density(self, X, grid_points):
        """Calcule la densité sur les points de la grille."""
        # Utiliser KDTree pour trouver les voisins dans un rayon de 2*sigma
        tree = KDTree(X)
        indices = tree.query_radius(grid_points, r=2*self.sigma)
        
        # Fonction noyau (Gaussian kernel)
        def kernel(dist_sq, sigma):
            return np.exp(-dist_sq / (2 * sigma**2))

        densities = np.zeros(len(grid_points))
        for i, idx in enumerate(indices):
            if len(idx) > 0:
                # Calculer la distance euclidienne au carré
                distances_sq = np.sum((X[idx] - grid_points[i])**2, axis=1)
                densities[i] = np.sum(kernel(distances_sq, self.sigma))
        
        return densities / len(X) # Normalisation par N
    
    def _find_dense_cells(self, grid_points, densities):
        """Identifie les cellules denses de la grille."""
        return grid_points[densities > self.xi]
    
    def _density_gradient_ascent(self, X, current):
        """Effectue une seule étape d'ascension de gradient."""
        # Calculer les poids (valeur du noyau) pour tous les points X
        distances_sq = np.sum((X - current)**2, axis=1)
        weights = np.exp(-distances_sq / (2 * self.sigma**2))
        
        sum_weights = np.sum(weights)
        
        if sum_weights == 0:
            return np.zeros_like(current), 0
            
        # Formule du gradient pour le noyau Gaussien:
        # gradient = (1/sum_weights) * sum_i [ (X_i - x) * weights_i / sigma^2 ]
        # La division par la somme des poids est dans la mise à jour 'current'
        
        # Terme de la somme : (X_i - x) * weights_i
        sum_term = np.sum((X - current) * weights[:, np.newaxis], axis=0)
        
        # Le gradient final est le "centre de gravité" pondéré (Mean Shift)
        # Mais nous allons utiliser la formule de mise à jour simplifiée de DENCLUE:
        # x_{k+1} = x_k + step_size * ( sum_i (X_i - x_k) * weights_i / sigma^2 )
        gradient = sum_term / (self.sigma**2)
        
        return gradient, sum_weights

    def fit(self, X):
        
        # Étape 1: Création de la grille et calcul des densités
        grid_points = self._create_grid(X)
        densities = self._compute_grid_density(X, grid_points)
        
        # Étape 2: Identification des points de départ denses
        dense_points = self._find_dense_cells(grid_points, densities)
        
        if len(dense_points) == 0:
            warnings.warn("Aucune cellule dense trouvée. Ajustez sigma ou xi.")
            self.labels_ = np.zeros(len(X), dtype=int) - 1
            self.attractors_ = np.array([])
            return self
        
        # Étape 3: Recherche des attracteurs par Mean Shift (Gradient Ascent)
        attractors = []
        attractor_map = {} # Map pour associer les points de la grille aux attracteurs
        
        for i, point in enumerate(dense_points):
            current = point.copy()
            
            for _ in range(self.max_iter):
                gradient, sum_weights = self._density_gradient_ascent(X, current)
                
                if sum_weights == 0 or np.linalg.norm(gradient) < 1e-6:
                    break
                    
                current = current + self.step_size * gradient
            
            # Vérifier si c'est un nouvel attracteur
            is_new = True
            for j, attractor in enumerate(attractors):
                # Les attracteurs sont considérés comme les mêmes s'ils sont proches
                if np.linalg.norm(current - attractor) < self.sigma: 
                    is_new = False
                    attractor_map[i] = j
                    break
            
            if is_new:
                attractors.append(current)
                attractor_map[i] = len(attractors) - 1
                
        self.attractors_ = np.array(attractors)
        
        # Étape 4: Assignation des points aux attracteurs
        # Chaque point X est assigné à l'attracteur le plus proche
        if len(self.attractors_) == 0:
             self.labels_ = np.zeros(len(X), dtype=int) - 1
             return self
             
        tree = KDTree(self.attractors_)
        distances, indices = tree.query(X, k=1)
        
        self.labels_ = indices.flatten()
        
        # Marquer comme bruit (-1) les points dont l'attracteur le plus proche 
        # est trop éloigné (par exemple, au-delà de 2*sigma)
        too_far = distances.flatten() > 2 * self.sigma
        self.labels_[too_far] = -1
        
        return self

    def fit_predict(self, X, y=None):
        return self.fit(X).labels_


# --- Fonction d'Exemple Simple ---

def example_denclue_and_dbscan():
    """Exécute un exemple simple de DENCLUE et DBSCAN pour la visualisation."""
    
    # 1. Préparation des données (Moons est un bon cas test)
    X, y = make_moons(n_samples=200, noise=0.08, random_state=42)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 2. DENCLUE
    denclue = DenclueOptimized(sigma=0.3, xi=0.01) # xi ajusté pour des données normalisées
    denclue_labels = denclue.fit_predict(X_scaled)
    
    # 3. DBSCAN pour la comparaison
    dbscan = DBSCAN(eps=0.3, min_samples=5)
    dbscan_labels = dbscan.fit_predict(X_scaled)
    
    # 4. Affichage des résultats
    
    print("\nDENCLUE: Nombre de clusters =", len(np.unique(denclue_labels[denclue_labels != -1])))
    print("DBSCAN: Nombre de clusters =", len(np.unique(dbscan_labels[dbscan_labels != -1])))
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    titles = ["Original Data", "DENCLUE Clustering", "DBSCAN Clustering"]
    labels_list = [y, denclue_labels, dbscan_labels]

    # 
    
    for ax, title, labels in zip(axes, titles, labels_list):
        unique_labels = np.unique(labels)
        colors = plt.cm.tab20(np.linspace(0, 1, len(unique_labels)))
        
        for k, col in zip(unique_labels, colors):
            label_name = f'Cluster {k}'
            if k == -1:
                col = 'gray'
                label_name = 'Bruit (-1)'
            
            class_mask = (labels == k)
            ax.scatter(X_scaled[class_mask, 0], X_scaled[class_mask, 1], 
                       c=[col], s=50, label=label_name, alpha=0.7)
        
        # Afficher les attracteurs DENCLUE
        if title == "DENCLUE Clustering" and denclue.attractors_ is not None:
             ax.scatter(denclue.attractors_[:, 0], denclue.attractors_[:, 1],
                        c='red', s=150, marker='X', edgecolors='black', label='Attracteurs')
        
        ax.set_title(title)
        ax.set_xlabel('Feature 1')
        ax.set_ylabel('Feature 2')
        # ax.legend()
        
    plt.tight_layout()
    plt.show()

# --- Exécution ---

if __name__ == "__main__":
    print("=== Exécution de l'exemple DENCLUE et DBSCAN ===")
    # J'ai remplacé l'appel à la fonction non définie 'example_denclue()' 
    # par l'appel à la fonction d'exemple corrigée
    example_denclue_and_dbscan()


"""
Graphique du Milieu : DENCLUE Clustering

Ce graphique montre le résultat de l'algorithme DENCLUE (DENsity-based CLUstEring) que vous avez implémenté.

    Principe : DENCLUE fonctionne en trouvant les attracteurs de densité (points rouges 'X') via une technique appelée ascension de gradient de densité. Chaque point de donnée est ensuite assigné à l'attracteur le plus proche, ou classé comme bruit.

    Les Attracteurs (Points Rouges 'X') : Les nombreux 'X' rouges représentent les pics de densité trouvés sur la grille.

    Le Résultat :

        L'algorithme a identifié plusieurs clusters (représentés par de nombreuses couleurs différentes : gris, vert, orange, marron, etc.).

        Problème d'Over-Clustering (Sur-Segmentation) : Le code semble avoir trouvé beaucoup plus de clusters que les deux lunes originales. Cela est dû au fait que la phase de liaison des attracteurs (non implémentée dans votre code simple) ou les paramètres (σ et ξ) n'ont pas permis de fusionner les attracteurs proches appartenant à la même structure de haute densité. Chaque petit pic de densité est devenu un cluster distinct.

        Points Gris : La grande majorité des points, en particulier ceux autour des attracteurs, sont assignés à un cluster (couleurs vives ou gris clair). Les points gris foncé au bord pourraient être le bruit (-1), mais la majorité des points sont classés.

    Conclusion DENCLUE : La version implémentée est très sensible et a découpé les deux grandes lunes en de multiples sous-clusters. Pour obtenir les deux lunes, il faudrait soit ajuster drastiquement le paramètre ξ (seuil de densité minimale) pour réduire le nombre d'attracteurs, soit ajouter l'étape de liaison des attracteurs (selon la définition originale de DENCLUE).
"""
