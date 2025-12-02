import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist, squareform

class SimpleCoClustering:
    """Co-clustering simple et pédagogique."""
    
    def __init__(self, n_row_clusters=2, n_col_clusters=2):
        self.n_row_clusters = n_row_clusters
        self.n_col_clusters = n_col_clusters
        self.row_labels_ = None
        self.col_labels_ = None
        
    def fit(self, X):
        """
        Applique le co-clustering à la matrice X.
        
        Parameters:
        -----------
        X : array 2D, matrice de données (lignes × colonnes)
        """
        self.X = np.array(X, dtype=float)
        n_rows, n_cols = self.X.shape
        
        print(f"Matrice de taille: {n_rows} × {n_cols}")
        print(f"Recherche de {self.n_row_clusters} clusters de lignes")
        print(f"Recherche de {self.n_col_clusters} clusters de colonnes")
        print("-" * 50)
        
        # Étape 1: Normalisation par ligne (centrage)
        print("Étape 1: Normalisation par ligne...")
        row_means = self.X.mean(axis=1, keepdims=True)
        X_row_norm = self.X - row_means
        
        # Étape 2: Clustering des lignes
        print("Étape 2: Clustering des lignes...")
        row_distances = pdist(X_row_norm, metric='euclidean')
        row_linkage = linkage(row_distances, method='average')
        
        # Découpage du dendrogramme pour obtenir k clusters
        from scipy.cluster.hierarchy import fcluster
        self.row_labels_ = fcluster(row_linkage, self.n_row_clusters, criterion='maxclust') - 1
        
        # Étape 3: Normalisation par colonne
        print("Étape 3: Normalisation par colonne...")
        col_means = self.X.mean(axis=0, keepdims=True)
        X_col_norm = self.X - col_means
        
        # Étape 4: Clustering des colonnes
        print("Étape 4: Clustering des colonnes...")
        col_distances = pdist(X_col_norm.T, metric='euclidean')
        col_linkage = linkage(col_distances, method='average')
        self.col_labels_ = fcluster(col_linkage, self.n_col_clusters, criterion='maxclust') - 1
        
        # Étape 5: Réorganisation
        print("Étape 5: Réorganisation de la matrice...")
        self._reorder_matrix()
        
        return self
    
    def _reorder_matrix(self):
        """Réorganise la matrice selon les clusters trouvés."""
        # Trier les lignes par cluster
        row_order = np.argsort(self.row_labels_)
        self.X_rows_sorted = self.X[row_order, :]
        self.row_labels_sorted = self.row_labels_[row_order]
        
        # Trier les colonnes par cluster
        col_order = np.argsort(self.col_labels_)
        self.X_sorted = self.X_rows_sorted[:, col_order]
        self.col_labels_sorted = self.col_labels_[col_order]
        
    def display_results(self, row_names=None, col_names=None):
        """Affiche les résultats de manière lisible."""
        if row_names is None:
            row_names = [f"Ligne{i}" for i in range(len(self.row_labels_))]
        if col_names is None:
            col_names = [f"Col{i}" for i in range(len(self.col_labels_))]
        
        print("\n" + "=" * 60)
        print("RÉSULTATS DU CO-CLUSTERING")
        print("=" * 60)
        
        # Clusters de lignes
        print("\nCLUSTERS DE LIGNES:")
        for cluster_id in range(self.n_row_clusters):
            indices = np.where(self.row_labels_ == cluster_id)[0]
            names = [row_names[i] for i in indices]
            print(f"  Cluster {cluster_id}: {', '.join(names)}")
        
        # Clusters de colonnes
        print("\nCLUSTERS DE COLONNES:")
        for cluster_id in range(self.n_col_clusters):
            indices = np.where(self.col_labels_ == cluster_id)[0]
            names = [col_names[i] for i in indices]
            print(f"  Cluster {cluster_id}: {', '.join(names)}")
        
        # Matrice réorganisée
        print("\nMATRICE RÉORGANISÉE:")
        print("-" * 40)
        
        # Obtenir l'ordre trié
        row_order = np.argsort(self.row_labels_)
        col_order = np.argsort(self.col_labels_)
        
        # Afficher les noms des colonnes triées
        sorted_col_names = [col_names[i] for i in col_order]
        header = " " * 10 + "  ".join([f"{name:>8}" for name in sorted_col_names])
        print(header)
        print("-" * (10 + 10 * len(sorted_col_names)))
        
        # Afficher chaque ligne avec son nom
        for i, row_idx in enumerate(row_order):
            row_name = row_names[row_idx]
            cluster_id = self.row_labels_[row_idx]
            values = self.X[row_idx, col_order]
            
            # Format: "Nom [cluster] | valeurs"
            row_str = f"{row_name} [{cluster_id}] | "
            row_str += "  ".join([f"{v:>8.2f}" for v in values])
            print(row_str)
            
            # Ligne de séparation entre clusters
            if i < len(row_order)-1 and self.row_labels_[row_order[i]] != self.row_labels_[row_order[i+1]]:
                print("-" * (10 + 10 * len(sorted_col_names)))

# Exemple avec nos données manuelles
def example_manual_coclustering():
    """Exemple simple de co-clustering avec nos données."""
    
    print("=" * 60)
    print("EXEMPLE MANUEL DE CO-CLUSTERING")
    print("=" * 60)
    
    # Nos données de films
    X = np.array([
        [5, 1, 3, 1],  # Alice
        [4, 2, 2, 2],  # Bob
        [1, 5, 2, 4],  # Claire
        [2, 4, 3, 5]   # David
    ], dtype=float)
    
    row_names = ["Alice", "Bob", "Claire", "David"]
    col_names = ["Action", "Romance", "Comédie", "Documentaire"]
    
    print("Matrice originale:")
    print("-" * 40)
    print("        " + "  ".join([f"{name:>12}" for name in col_names]))
    for i, name in enumerate(row_names):
        print(f"{name:8} " + "  ".join([f"{v:>12.1f}" for v in X[i]]))
    
    # Appliquer le co-clustering
    cocluster = SimpleCoClustering(n_row_clusters=2, n_col_clusters=3)
    cocluster.fit(X)
    
    # Afficher les résultats
    cocluster.display_results(row_names=row_names, col_names=col_names)
    
    # Visualisation
    visualize_coclustering(X, cocluster, row_names, col_names)
    
    return cocluster

def visualize_coclustering(X, cocluster, row_names, col_names):
    """Visualise les résultats du co-clustering."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Matrice originale
    im1 = axes[0].imshow(X, cmap='YlOrRd', aspect='auto')
    axes[0].set_title('Matrice originale')
    axes[0].set_xlabel('Colonnes')
    axes[0].set_ylabel('Lignes')
    axes[0].set_xticks(range(len(col_names)))
    axes[0].set_xticklabels(col_names, rotation=45, ha='right')
    axes[0].set_yticks(range(len(row_names)))
    axes[0].set_yticklabels(row_names)
    plt.colorbar(im1, ax=axes[0])
    
    # Matrice réorganisée
    # Obtenir l'ordre trié
    row_order = np.argsort(cocluster.row_labels_)
    col_order = np.argsort(cocluster.col_labels_)
    
    X_sorted = X[row_order, :][:, col_order]
    
    im2 = axes[1].imshow(X_sorted, cmap='YlOrRd', aspect='auto')
    axes[1].set_title('Matrice réorganisée par co-clustering')
    axes[1].set_xlabel('Colonnes (réordonnées)')
    axes[1].set_ylabel('Lignes (réordonnées)')
    
    # Afficher les noms triés
    sorted_row_names = [row_names[i] for i in row_order]
    sorted_col_names = [col_names[i] for i in col_order]
    
    axes[1].set_xticks(range(len(sorted_col_names)))
    axes[1].set_xticklabels(sorted_col_names, rotation=45, ha='right')
    axes[1].set_yticks(range(len(sorted_row_names)))
    axes[1].set_yticklabels(sorted_row_names)
    
    # Ajouter des lignes de séparation entre clusters
    # Pour les lignes
    row_cluster_changes = np.where(np.diff(cocluster.row_labels_[row_order]) != 0)[0]
    for change in row_cluster_changes:
        axes[1].axhline(y=change+0.5, color='blue', linewidth=2, linestyle='--')
    
    # Pour les colonnes
    col_cluster_changes = np.where(np.diff(cocluster.col_labels_[col_order]) != 0)[0]
    for change in col_cluster_changes:
        axes[1].axvline(x=change+0.5, color='blue', linewidth=2, linestyle='--')
    
    plt.colorbar(im2, ax=axes[1])
    plt.tight_layout()
    plt.show()
    
    # Afficher les blocs identifiés
    print("\n" + "=" * 60)
    print("BLOCS HOMOGÈNES IDENTIFIÉS")
    print("=" * 60)
    
    # Calculer la moyenne de chaque bloc
    for row_cluster in range(cocluster.n_row_clusters):
        for col_cluster in range(cocluster.n_col_clusters):
            # Indices des lignes dans ce cluster
            row_indices = np.where(cocluster.row_labels_ == row_cluster)[0]
            # Indices des colonnes dans ce cluster
            col_indices = np.where(cocluster.col_labels_ == col_cluster)[0]
            
            # Extraire le bloc
            block = X[np.ix_(row_indices, col_indices)]
            
            if block.size > 0:
                block_mean = np.mean(block)
                block_std = np.std(block)
                
                row_names_list = [row_names[i] for i in row_indices]
                col_names_list = [col_names[i] for i in col_indices]
                
                print(f"\nBloc (Lignes {row_cluster} × Colonnes {col_cluster}):")
                print(f"  Lignes: {', '.join(row_names_list)}")
                print(f"  Colonnes: {', '.join(col_names_list)}")
                print(f"  Valeurs moyennes: {block_mean:.2f} ± {block_std:.2f}")
                print(f"  Plage des valeurs: {block.min():.1f} à {block.max():.1f}")

# Version 2 : Algorithme plus avancé (inspiré de Cheng & Church)
class ChengChurchCoClustering:
    """Implémentation simplifiée de l'algorithme de Cheng & Church."""
    
    def __init__(self, n_clusters=3, max_iter=100, threshold=1.0):
        self.n_clusters = n_clusters  # Même nombre pour lignes et colonnes
        self.max_iter = max_iter
        self.threshold = threshold
        self.row_labels_ = None
        self.col_labels_ = None
        
    def fit(self, X):
        """Applique l'algorithme de co-clustering de Cheng & Church."""
        X = np.array(X, dtype=float)
        n_rows, n_cols = X.shape
        
        print(f"Algorithme de Cheng & Church")
        print(f"Matrice: {n_rows} × {n_cols}, Clusters: {self.n_clusters}")
        print("-" * 50)
        
        # Initialisation aléatoire
        self.row_labels_ = np.random.randint(0, self.n_clusters, n_rows)
        self.col_labels_ = np.random.randint(0, self.n_clusters, n_cols)
        
        # Itérations d'amélioration
        for iteration in range(self.max_iter):
            print(f"Iteration {iteration+1}", end="")
            
            # Étape 1: Optimiser les clusters de lignes
            old_row_labels = self.row_labels_.copy()
            for i in range(n_rows):
                best_cluster = self._find_best_cluster_for_row(X, i)
                self.row_labels_[i] = best_cluster
            
            # Étape 2: Optimiser les clusters de colonnes
            old_col_labels = self.col_labels_.copy()
            for j in range(n_cols):
                best_cluster = self._find_best_cluster_for_col(X, j)
                self.col_labels_[j] = best_cluster
            
            # Vérifier la convergence
            row_changed = np.sum(old_row_labels != self.row_labels_)
            col_changed = np.sum(old_col_labels != self.col_labels_)
            
            print(f" - Lignes changées: {row_changed}, Colonnes changées: {col_changed}")
            
            if row_changed == 0 and col_changed == 0:
                print(f"Convergence atteinte après {iteration+1} itérations")
                break
        
        return self
    
    def _find_best_cluster_for_row(self, X, row_idx):
        """Trouve le meilleur cluster pour une ligne donnée."""
        best_cluster = self.row_labels_[row_idx]
        best_score = float('inf')
        
        for cluster in range(self.n_clusters):
            # Calculer le score si on mettait cette ligne dans ce cluster
            score = self._calculate_row_score(X, row_idx, cluster)
            
            if score < best_score:
                best_score = score
                best_cluster = cluster
        
        return best_cluster
    
    def _find_best_cluster_for_col(self, X, col_idx):
        """Trouve le meilleur cluster pour une colonne donnée."""
        best_cluster = self.col_labels_[col_idx]
        best_score = float('inf')
        
        for cluster in range(self.n_clusters):
            score = self._calculate_col_score(X, col_idx, cluster)
            
            if score < best_score:
                best_score = score
                best_cluster = cluster
        
        return best_cluster
    
    def _calculate_row_score(self, X, row_idx, cluster):
        """Calcule le score pour une ligne dans un cluster."""
        # Trouver toutes les lignes dans ce cluster
        rows_in_cluster = np.where(self.row_labels_ == cluster)[0]
        
        # Trouver les colonnes dans chaque cluster de colonnes
        score = 0
        for col_cluster in range(self.n_clusters):
            cols_in_cluster = np.where(self.col_labels_ == col_cluster)[0]
            
            if len(cols_in_cluster) == 0:
                continue
            
            # Extraire le bloc
            block_rows = list(rows_in_cluster) + [row_idx]
            block = X[np.ix_(block_rows, cols_in_cluster)]
            
            # Calculer la variance résiduelle
            score += self._calculate_residue(block)
        
        return score
    
    def _calculate_col_score(self, X, col_idx, cluster):
        """Calcule le score pour une colonne dans un cluster."""
        cols_in_cluster = np.where(self.col_labels_ == cluster)[0]
        
        score = 0
        for row_cluster in range(self.n_clusters):
            rows_in_cluster = np.where(self.row_labels_ == row_cluster)[0]
            
            if len(rows_in_cluster) == 0:
                continue
            
            # Extraire le bloc
            block_cols = list(cols_in_cluster) + [col_idx]
            block = X[np.ix_(rows_in_cluster, block_cols)]
            
            score += self._calculate_residue(block)
        
        return score
    
    def _calculate_residue(self, block):
        """Calcule le résidu d'un bloc (simplifié)."""
        if block.size == 0:
            return 0
        
        row_means = block.mean(axis=1, keepdims=True)
        col_means = block.mean(axis=0, keepdims=True)
        grand_mean = block.mean()
        
        residue = np.sum((block - row_means - col_means + grand_mean)**2)
        return residue / block.size if block.size > 0 else 0

# Exemple 2 : Données génétiques (expression génique)
def gene_expression_example():
    """Exemple de co-clustering sur des données d'expression génique."""
    
    print("\n" + "=" * 60)
    print("EXEMPLE: DONNÉES D'EXPRESSION GÉNIQUE")
    print("=" * 60)
    
    # Simuler des données d'expression génique
    np.random.seed(42)
    
    # 10 gènes, 8 conditions expérimentales
    n_genes = 10
    n_conditions = 8
    
    # Créer des patterns de co-expression
    X = np.random.normal(0, 1, (n_genes, n_conditions))
    
    # Ajouter des patterns de co-expression
    # Groupe 1: Gènes 0-3, Conditions 0-3 (expression élevée)
    X[0:4, 0:4] += 3.0
    
    # Groupe 2: Gènes 4-6, Conditions 4-6 (expression moyenne)
    X[4:7, 4:7] += 1.5
    
    # Groupe 3: Gènes 7-9, Conditions 4-7 (expression faible)
    X[7:10, 4:8] -= 2.0
    
    # Ajouter du bruit
    X += np.random.normal(0, 0.5, X.shape)
    
    # Noms
    gene_names = [f"Gène_{i}" for i in range(n_genes)]
    condition_names = [f"Cond_{i}" for i in range(n_conditions)]
    
    print("Matrice d'expression génique simulée:")
    print(f"Dimensions: {n_genes} gènes × {n_conditions} conditions")
    print("\n5 premiers gènes, 5 premières conditions:")
    print(X[:5, :5].round(2))
    
    # Appliquer le co-clustering
    cocluster = SimpleCoClustering(n_row_clusters=3, n_col_clusters=3)
    cocluster.fit(X)
    
    # Afficher les résultats
    print("\n" + "=" * 50)
    print("RÉSULTATS DU CO-CLUSTERING GÉNÉTIQUE")
    print("=" * 50)
    
    # Afficher les clusters de gènes
    print("\nClusters de gènes (groupes de gènes co-exprimés):")
    for cluster_id in range(cocluster.n_row_clusters):
        indices = np.where(cocluster.row_labels_ == cluster_id)[0]
        genes = [gene_names[i] for i in indices]
        print(f"  Cluster {cluster_id} ({len(genes)} gènes): {', '.join(genes)}")
    
    # Afficher les clusters de conditions
    print("\nClusters de conditions (conditions similaires):")
    for cluster_id in range(cocluster.n_col_clusters):
        indices = np.where(cocluster.col_labels_ == cluster_id)[0]
        conditions = [condition_names[i] for i in indices]
        print(f"  Cluster {cluster_id} ({len(conditions)} conditions): {', '.join(conditions)}")
    
    # Visualisation
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Heatmap originale
    im1 = axes[0].imshow(X, cmap='RdYlBu_r', aspect='auto')
    axes[0].set_title('Expression génique originale')
    axes[0].set_xlabel('Conditions expérimentales')
    axes[0].set_ylabel('Gènes')
    axes[0].set_xticks(range(n_conditions))
    axes[0].set_xticklabels(condition_names, rotation=45, ha='right', fontsize=8)
    axes[0].set_yticks(range(n_genes))
    axes[0].set_yticklabels(gene_names, fontsize=8)
    plt.colorbar(im1, ax=axes[0], label="Niveau d'expression")
    
    # Heatmap réorganisée
    row_order = np.argsort(cocluster.row_labels_)
    col_order = np.argsort(cocluster.col_labels_)
    X_sorted = X[row_order, :][:, col_order]
    
    im2 = axes[1].imshow(X_sorted, cmap='RdYlBu_r', aspect='auto')
    axes[1].set_title('Expression après co-clustering')
    axes[1].set_xlabel('Conditions (réordonnées)')
    axes[1].set_ylabel('Gènes (réordonnées)')
    
    # Noms triés
    sorted_gene_names = [gene_names[i] for i in row_order]
    sorted_cond_names = [condition_names[i] for i in col_order]
    
    axes[1].set_xticks(range(len(sorted_cond_names)))
    axes[1].set_xticklabels(sorted_cond_names, rotation=45, ha='right', fontsize=8)
    axes[1].set_yticks(range(len(sorted_gene_names)))
    axes[1].set_yticklabels(sorted_gene_names, fontsize=8)
    
    # Ajouter les séparations de clusters
    row_changes = np.where(np.diff(cocluster.row_labels_[row_order]) != 0)[0]
    for change in row_changes:
        axes[1].axhline(y=change+0.5, color='black', linewidth=2)
    
    col_changes = np.where(np.diff(cocluster.col_labels_[col_order]) != 0)[0]
    for change in col_changes:
        axes[1].axvline(x=change+0.5, color='black', linewidth=2)
    
    plt.colorbar(im2, ax=axes[1], label="Niveau d'expression")
    plt.tight_layout()
    plt.show()
    
    # Interprétation biologique
    print("\n" + "=" * 50)
    print("INTERPRÉTATION BIOLOGIQUE")
    print("=" * 50)
    
    for row_cluster in range(cocluster.n_row_clusters):
        for col_cluster in range(cocluster.n_col_clusters):
            row_indices = np.where(cocluster.row_labels_ == row_cluster)[0]
            col_indices = np.where(cocluster.col_labels_ == col_cluster)[0]
            
            block = X[np.ix_(row_indices, col_indices)]
            
            if block.size > 0:
                block_mean = np.mean(block)
                
                if block_mean > 1.0:
                    level = "FORTEMENT EXPRIMÉ"
                elif block_mean > 0:
                    level = "EXPRIMÉ"
                elif block_mean > -1.0:
                    level = "FAIBLEMENT EXPRIMÉ"
                else:
                    level = "RÉPRIMÉ"
                
                gene_list = [gene_names[i] for i in row_indices[:3]]  # 3 premiers
                cond_list = [condition_names[i] for i in col_indices[:3]]  # 3 premières
                
                if len(gene_list) > 0 and len(cond_list) > 0:
                    print(f"\n• Gènes {', '.join(gene_list)}...")
                    print(f"  sont {level} dans les conditions {', '.join(cond_list)}...")
                    print(f"  (moyenne: {block_mean:.2f})")
    
    return cocluster, X, gene_names, condition_names

# Exemple 3 : Recommandation avec co-clustering
def recommendation_example():
    """Exemple de système de recommandation avec co-clustering."""
    
    print("\n" + "=" * 60)
    print("EXEMPLE: SYSTÈME DE RECOMMANDATION")
    print("=" * 60)
    
    # Simuler des données utilisateur-produit
    np.random.seed(123)
    
    # 20 utilisateurs, 15 produits
    n_users = 20
    n_products = 15
    
    # Créer des groupes d'utilisateurs avec préférences similaires
    ratings = np.zeros((n_users, n_products))
    
    # Groupe 1: Utilisateurs 0-6, Produits 0-4 (notes élevées)
    ratings[0:7, 0:5] = np.random.uniform(4, 5, (7, 5))
    
    # Groupe 2: Utilisateurs 7-13, Produits 5-9 (notes élevées)
    ratings[7:14, 5:10] = np.random.uniform(4, 5, (7, 5))
    
    # Groupe 3: Utilisateurs 14-19, Produits 10-14 (notes élevées)
    ratings[14:20, 10:15] = np.random.uniform(4, 5, (6, 5))
    
    # Notes aléatoires pour le reste
    ratings += np.random.uniform(0, 2, ratings.shape)
    
    # Mettre des NaN pour les produits non notés (30% manquants)
    mask = np.random.random(ratings.shape) < 0.3
    ratings_with_nan = ratings.copy()
    ratings_with_nan[mask] = np.nan
    
    # Noms
    user_names = [f"User_{i}" for i in range(n_users)]
    product_names = [f"Produit_{chr(65+i)}" for i in range(n_products)]  # A, B, C...
    
    print("Matrice utilisateur-produit (notes 1-5):")
    print(f"Dimensions: {n_users} utilisateurs × {n_products} produits")
    print(f"Données manquantes: {np.sum(mask)}/{ratings.size} ({np.sum(mask)/ratings.size*100:.1f}%)")
    
    # Pour le co-clustering, remplacer NaN par la moyenne
    ratings_filled = np.where(np.isnan(ratings_with_nan), 
                              np.nanmean(ratings_with_nan, axis=0), 
                              ratings_with_nan)
    
    # Appliquer le co-clustering
    cocluster = SimpleCoClustering(n_row_clusters=3, n_col_clusters=3)
    cocluster.fit(ratings_filled)
    
    # Recommandations pour un utilisateur
    target_user = 2  # User_2
    target_user_cluster = cocluster.row_labels_[target_user]
    
    print(f"\n" + "=" * 50)
    print(f"RECOMMANDATIONS POUR {user_names[target_user]}")
    print("=" * 50)
    
    print(f"\n{user_names[target_user]} est dans le cluster {target_user_cluster}")
    print(f"Avec les utilisateurs: ", end="")
    
    # Trouver les utilisateurs du même cluster
    same_cluster_users = np.where(cocluster.row_labels_ == target_user_cluster)[0]
    same_cluster_users = same_cluster_users[same_cluster_users != target_user]  # Exclure l'utilisateur cible
    
    user_list = [user_names[i] for i in same_cluster_users[:5]]  # 5 premiers
    print(", ".join(user_list))
    
    # Produits aimés par ces utilisateurs
    print("\nProduits aimés par ce groupe:")
    
    # Trouver les produits du cluster de produits associé
    product_clusters_for_group = []
    for user in same_cluster_users:
        # Regarder quels produits cet utilisateur a bien notés
        user_ratings = ratings_filled[user]
        good_products = np.where(user_ratings > 3.5)[0]
        
        for prod in good_products:
            product_clusters_for_group.append(cocluster.col_labels_[prod])
    
    if product_clusters_for_group:
        # Cluster de produits le plus fréquent
        from collections import Counter
        cluster_counter = Counter(product_clusters_for_group)
        most_common_cluster = cluster_counter.most_common(1)[0][0]
        
        print(f"  Cluster de produits préféré: {most_common_cluster}")
        
        # Produits dans ce cluster
        products_in_cluster = np.where(cocluster.col_labels_ == most_common_cluster)[0]
        
        # Produits que l'utilisateur n'a pas encore bien notés
        user_ratings = ratings_filled[target_user]
        products_to_recommend = []
        
        for prod in products_in_cluster:
            if user_ratings[prod] < 3.0:  # Pas encore bien noté
                # Estimer la note basée sur les utilisateurs similaires
                similar_users_ratings = ratings_filled[same_cluster_users, prod]
                predicted_rating = np.nanmean(similar_users_ratings)
                
                if predicted_rating > 3.5:
                    products_to_recommend.append((prod, predicted_rating))
        
        # Trier par note prédite
        products_to_recommend.sort(key=lambda x: x[1], reverse=True)
        
        print(f"\nRecommandations pour {user_names[target_user]}:")
        for prod_idx, pred_rating in products_to_recommend[:5]:  # Top 5
            prod_name = product_names[prod_idx]
            print(f"  • {prod_name}: note prédite {pred_rating:.2f}/5")
    else:
        print("  Pas assez de données pour faire des recommandations")
    
    return cocluster, ratings_filled, user_names, product_names

# Fonction principale
def main():
    """Exécute tous les exemples de co-clustering."""
    print("=" * 60)
    print("CO-CLUSTERING (SIMULTANEOUS CLUSTERING)")
    print("=" * 60)
    
    # Exemple 1 : Pédagogique (films)
    print("\n1. EXEMPLE PÉDAGOGIQUE: NOTES DE FILMS")
    cocluster_films = example_manual_coclustering()
    
    # Exemple 2 : Données génétiques
    print("\n2. APPLICATION: EXPRESSION GÉNIQUE")
    cocluster_genes, X_genes, gene_names, cond_names = gene_expression_example()
    
    # Exemple 3 : Recommandation
    print("\n3. APPLICATION: SYSTÈME DE RECOMMANDATION")
    cocluster_rec, ratings, user_names, prod_names = recommendation_example()
    
    print("\n" + "=" * 60)
    print("CONCLUSION SUR LE CO-CLUSTERING")
    print("=" * 60)
    print("""
    Le co-clustering permet de découvrir simultanément:
    
    1. GROUPES DE LIGNES ayant des comportements similaires
    2. GROUPES DE COLONNES présentant des patterns similaires
    3. BLOCS HOMOGÈNES dans la matrice réorganisée
    
    Applications:
    - Biologie: Groupes de gènes co-exprimés dans certaines conditions
    - Marketing: Segments de clients avec préférences similaires pour certains produits
    - Text mining: Groupes de documents partageant certains mots-clés
    - Recommandation: Découverte de groupes utilisateur-produit
    
    Avantages vs clustering simple:
    - Découverte de structures bidimensionnelles
    - Meilleure interprétabilité
    - Réduction du bruit par regroupement bidirectionnel
    
    Algorithmes populaires:
    1. Cheng & Church (basé sur la variance résiduelle)
    2. Spectral Co-clustering (basé sur la décomposition en valeurs singulières)
    3. Information-Theoretic Co-clustering
    """)

if __name__ == "__main__":
    main()
