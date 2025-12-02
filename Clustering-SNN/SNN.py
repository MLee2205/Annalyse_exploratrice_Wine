import numpy as np
from sklearn.neighbors import NearestNeighbors
import networkx as nx
import matplotlib.pyplot as plt

class SNNClustering:
    """Clustering basé sur Shared Nearest Neighbor (SNN)."""
    
    def __init__(self, k=5, similarity_threshold=3, min_cluster_size=2):
        """
        Parameters:
        -----------
        k : int, nombre de plus proches voisins à considérer
        similarity_threshold : int, seuil minimum de voisins communs pour connecter deux points
        min_cluster_size : int, taille minimale d'un cluster
        """
        self.k = k
        self.similarity_threshold = similarity_threshold
        self.min_cluster_size = min_cluster_size
        self.labels_ = None
        self.snn_matrix_ = None
        
    def fit_predict(self, X):
        """
        Applique le clustering SNN aux données.
        
        Parameters:
        -----------
        X : array 2D de forme (n_samples, n_features)
        
        Returns:
        --------
        labels : array, labels des clusters (-1 pour le bruit)
        """
        n_samples = X.shape[0]
        
        # Étape 1 : Trouver les k plus proches voisins
        nn = NearestNeighbors(n_neighbors=self.k + 1)  # +1 pour inclure le point lui-même
        nn.fit(X)
        
        # Indices des k+1 plus proches voisins (inclut le point lui-même)
        distances, indices = nn.kneighbors(X)
        
        # Étape 2 : Calculer la matrice de similarité SNN
        snn_matrix = np.zeros((n_samples, n_samples), dtype=int)
        
        for i in range(n_samples):
            neighbors_i = set(indices[i])  # Ensemble des voisins de i (inclut i)
            
            for j in range(i + 1, n_samples):
                neighbors_j = set(indices[j])  # Ensemble des voisins de j (inclut j)
                
                # Nombre de voisins communs (intersection)
                common_neighbors = len(neighbors_i.intersection(neighbors_j))
                snn_matrix[i, j] = common_neighbors
                snn_matrix[j, i] = common_neighbors  # Symétrique
        
        self.snn_matrix_ = snn_matrix
        
        # Étape 3 : Construire le graphe SNN
        G = nx.Graph()
        
        # Ajouter les nœuds
        for i in range(n_samples):
            G.add_node(i)
        
        # Ajouter les arêtes si similarité ≥ seuil
        for i in range(n_samples):
            for j in range(i + 1, n_samples):
                if snn_matrix[i, j] >= self.similarity_threshold:
                    G.add_edge(i, j, weight=snn_matrix[i, j])
        
        # Étape 4 : Trouver les composantes connexes (clusters)
        labels = -np.ones(n_samples, dtype=int)  # -1 pour le bruit
        cluster_id = 0
        
        for component in nx.connected_components(G):
            if len(component) >= self.min_cluster_size:
                for node in component:
                    labels[node] = cluster_id
                cluster_id += 1
        
        self.labels_ = labels
        return labels
    
    def plot_results(self, X, title="Clustering SNN"):
        """Visualise les résultats du clustering."""
        if self.labels_ is None:
            raise ValueError("Le modèle doit être entraîné d'abord")
        
        plt.figure(figsize=(10, 6))
        
        # Créer une palette de couleurs
        unique_labels = np.unique(self.labels_)
        n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
        
        colors = plt.cm.tab10(np.linspace(0, 1, max(10, n_clusters)))
        
        for label in unique_labels:
            if label == -1:
                # Bruit
                mask = self.labels_ == label
                plt.scatter(X[mask, 0], X[mask, 1], 
                           c='gray', s=50, alpha=0.5, 
                           label='Bruit' if np.any(mask) else None)
            else:
                # Cluster
                mask = self.labels_ == label
                plt.scatter(X[mask, 0], X[mask, 1], 
                           c=[colors[label % len(colors)]], 
                           s=100, alpha=0.7,
                           label=f'Cluster {label}')
        
        plt.title(f"{title} (k={self.k}, seuil={self.similarity_threshold})")
        plt.xlabel("Feature 1")
        plt.ylabel("Feature 2")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
    
    def get_snn_similarity(self, i, j):
        """Retourne la similarité SNN entre deux points."""
        if self.snn_matrix_ is None:
            raise ValueError("Le modèle doit être entraîné d'abord")
        return self.snn_matrix_[i, j]

# Exemple avec nos données manuelles
def example_snn_simple():
    """Exemple simple avec nos données de l'exemple manuel."""
    print("=" * 60)
    print("EXEMPLE SIMPLE SNN CLUSTERING")
    print("=" * 60)
    
    # Données de l'exemple manuel
    X = np.array([
        [1, 1],   # A
        [2, 2],   # B
        [3, 3],   # C
        [8, 8],   # D
        [9, 9],   # E
        [5, 5]    # F (point intermédiaire)
    ])
    
    noms = ['A', 'B', 'C', 'D', 'E', 'F']
    
    print(f"\nDonnées ({len(X)} points):")
    for i, (point, nom) in enumerate(zip(X, noms)):
        print(f"  {nom}{point}", end="  " if (i+1) % 3 != 0 else "\n")
    
    # Appliquer SNN Clustering
    snn = SNNClustering(k=3, similarity_threshold=3, min_cluster_size=2)
    labels = snn.fit_predict(X)
    
    print(f"\nParamètres: k={snn.k}, seuil={snn.similarity_threshold}")
    print("\nRésultats du clustering:")
    for i, (nom, label) in enumerate(zip(noms, labels)):
        cluster_name = f"Cluster {label}" if label != -1 else "Bruit"
        print(f"  {nom} → {cluster_name}")
    
    # Afficher quelques similarités SNN
    print(f"\nQuelques similarités SNN:")
    print(f"  Similarité(A,B) = {snn.get_snn_similarity(0, 1)}")
    print(f"  Similarité(C,F) = {snn.get_snn_similarity(2, 5)}")
    print(f"  Similarité(D,E) = {snn.get_snn_similarity(3, 4)}")
    print(f"  Similarité(A,D) = {snn.get_snn_similarity(0, 3)}")
    
    # Visualiser
    snn.plot_results(X, title="Exemple SNN - Données simples")
    
    return snn, labels

# Exemple 2 : Données avec clusters de densités différentes
def example_variable_density():
    """Exemple avec clusters de densités différentes."""
    print("\n" + "=" * 60)
    print("SNN SUR CLUSTERS DE DENSITÉS DIFFÉRENTES")
    print("=" * 60)
    
    np.random.seed(42)
    
    # Cluster 1 : Dense, petit
    cluster1 = np.random.randn(30, 2) * 0.3 + np.array([2, 2])
    
    # Cluster 2 : Sparse, grand
    cluster2 = np.random.randn(50, 2) * 1.5 + np.array([10, 10])
    
    # Bruit
    noise = np.random.rand(10, 2) * 15
    
    X = np.vstack([cluster1, cluster2, noise])
    
    # DBSCAN traditionnel échouerait ici (eps fixe)
    # Mais SNN devrait bien fonctionner
    
    # Appliquer SNN
    snn = SNNClustering(k=10, similarity_threshold=5, min_cluster_size=5)
    labels = snn.fit_predict(X)
    
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = np.sum(labels == -1)
    
    print(f"\nRésultats:")
    print(f"  Points totaux: {len(X)}")
    print(f"  Clusters trouvés: {n_clusters}")
    print(f"  Points considérés comme bruit: {n_noise}")
    
    # Visualiser
    plt.figure(figsize=(12, 5))
    
    # Données originales
    plt.subplot(1, 2, 1)
    plt.scatter(X[:30, 0], X[:30, 1], c='blue', s=50, alpha=0.7, label='Cluster dense')
    plt.scatter(X[30:80, 0], X[30:80, 1], c='green', s=50, alpha=0.7, label='Cluster sparse')
    plt.scatter(X[80:, 0], X[80:, 1], c='gray', s=50, alpha=0.5, label='Bruit')
    plt.title("Données originales")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Résultats SNN
    plt.subplot(1, 2, 2)
    unique_labels = np.unique(labels)
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
    
    for i, label in enumerate(unique_labels):
        if label == -1:
            mask = labels == label
            plt.scatter(X[mask, 0], X[mask, 1], c='gray', s=50, alpha=0.5, label='Bruit')
        else:
            mask = labels == label
            plt.scatter(X[mask, 0], X[mask, 1], c=[colors[i]], s=50, alpha=0.7, label=f'Cluster {label}')
    
    plt.title(f"Clustering SNN (k={snn.k}, seuil={snn.similarity_threshold})")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return snn, labels

# Exemple 3 : Comparaison SNN vs DBSCAN
def compare_snn_dbscan():
    """Compare SNN clustering avec DBSCAN traditionnel."""
    from sklearn.cluster import DBSCAN
    from sklearn.datasets import make_moons, make_circles
    
    print("\n" + "=" * 60)
    print("COMPARAISON SNN vs DBSCAN")
    print("=" * 60)
    
    # Créer des données complexes
    X1, _ = make_moons(n_samples=150, noise=0.05, random_state=42)
    X2, _ = make_circles(n_samples=150, factor=0.5, noise=0.05, random_state=42)
    X2 = X2 * 2 + np.array([3, 0])  # Déplacer le cercle
    
    X = np.vstack([X1, X2])
    
    # Appliquer SNN
    snn = SNNClustering(k=15, similarity_threshold=10, min_cluster_size=10)
    snn_labels = snn.fit_predict(X)
    
    # Appliquer DBSCAN
    dbscan = DBSCAN(eps=0.2, min_samples=5)
    dbscan_labels = dbscan.fit_predict(X)
    
    # Calculer les métriques
    snn_clusters = len(set(snn_labels)) - (1 if -1 in snn_labels else 0)
    dbscan_clusters = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
    
    print(f"\nRésultats:")
    print(f"  SNN: {snn_clusters} clusters, {np.sum(snn_labels == -1)} points bruit")
    print(f"  DBSCAN: {dbscan_clusters} clusters, {np.sum(dbscan_labels == -1)} points bruit")
    
    # Visualiser
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Données originales
    axes[0].scatter(X[:, 0], X[:, 1], c='blue', s=30, alpha=0.7)
    axes[0].set_title("Données originales")
    axes[0].set_xlabel("Feature 1")
    axes[0].set_ylabel("Feature 2")
    axes[0].grid(True, alpha=0.3)
    
    # SNN
    unique_snn = np.unique(snn_labels)
    colors_snn = plt.cm.tab10(np.linspace(0, 1, len(unique_snn)))
    
    for i, label in enumerate(unique_snn):
        if label == -1:
            mask = snn_labels == label
            axes[1].scatter(X[mask, 0], X[mask, 1], c='gray', s=30, alpha=0.5)
        else:
            mask = snn_labels == label
            axes[1].scatter(X[mask, 0], X[mask, 1], c=[colors_snn[i]], s=30, alpha=0.7)
    
    axes[1].set_title(f"SNN Clustering\nk={snn.k}, seuil={snn.similarity_threshold}")
    axes[1].set_xlabel("Feature 1")
    axes[1].set_ylabel("Feature 2")
    axes[1].grid(True, alpha=0.3)
    
    # DBSCAN
    unique_dbscan = np.unique(dbscan_labels)
    colors_dbscan = plt.cm.tab10(np.linspace(0, 1, len(unique_dbscan)))
    
    for i, label in enumerate(unique_dbscan):
        if label == -1:
            mask = dbscan_labels == label
            axes[2].scatter(X[mask, 0], X[mask, 1], c='gray', s=30, alpha=0.5)
        else:
            mask = dbscan_labels == label
            axes[2].scatter(X[mask, 0], X[mask, 1], c=[colors_dbscan[i]], s=30, alpha=0.7)
    
    axes[2].set_title(f"DBSCAN\neps={dbscan.eps}, min_samples={dbscan.min_samples}")
    axes[2].set_xlabel("Feature 1")
    axes[2].set_ylabel("Feature 2")
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return snn, dbscan, snn_labels, dbscan_labels

# Exemple 4 : SNN pour données textuelles (simplifié)
def snn_text_example():
    """Exemple simplifié de SNN pour des données textuelles."""
    print("\n" + "=" * 60)
    print("SNN POUR DONNÉES TEXTUELLES (SIMPLIFIÉ)")
    print("=" * 60)
    
    # Documents simulés (chaque ligne est un "document")
    documents = [
        "chat chien animal domestique",
        "chien fidèle compagnon",
        "chat indépendant félin",
        "ordinateur programme logiciel",
        "python java programmation",
        "algorithmie structure données",
        "voiture automobile transport",
        "moteur véhicule roue",
        "chat programme ordinateur",  # Document mixte!
    ]
    
    # Créer un vocabulaire simple
    from sklearn.feature_extraction.text import CountVectorizer
    
    vectorizer = CountVectorizer(binary=True)  # Présence/absence
    X = vectorizer.fit_transform(documents).toarray()
    
    print(f"\nDocuments ({len(documents)}):")
    for i, doc in enumerate(documents):
        print(f"  D{i}: {doc[:30]}...")
    
    print(f"\nVocabulaire ({len(vectorizer.vocabulary_)} mots):")
    print(f"  {list(vectorizer.vocabulary_.keys())}")
    
    # Convertir en array numpy pour SNN
    X_dense = X.astype(float)
    
    # Appliquer SNN
    snn = SNNClustering(k=3, similarity_threshold=2, min_cluster_size=2)
    labels = snn.fit_predict(X_dense)
    
    print(f"\nClusters trouvés:")
    clusters = {}
    for i, label in enumerate(labels):
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(f"D{i}")
    
    for label, docs in sorted(clusters.items()):
        cluster_name = f"Cluster {label}" if label != -1 else "Bruit"
        print(f"  {cluster_name}: {', '.join(docs)}")
    
    # Analyser les similarités pour le document mixte D8
    print(f"\nAnalyse du document mixte D8 (chat programme ordinateur):")
    for i in range(len(documents)):
        if i != 8:
            sim = snn.get_snn_similarity(8, i)
            print(f"  Similarité(D8, D{i}) = {sim}")
    
    return snn, labels, documents

# Exemple 5 : Version ultra-simplifiée (sans dépendances lourdes)
def ultra_simple_snn():
    """Version ultra-simplifiée de SNN pour comprendre l'algorithme."""
    print("\n" + "=" * 60)
    print("VERSION ULTRA-SIMPLIFIÉE DE SNN")
    print("=" * 60)
    
    # Données très simples
    points = [
        [1, 1],  # Groupe A
        [1, 2],  # Groupe A
        [1, 3],  # Groupe A
        [8, 8],  # Groupe B
        [8, 9],  # Groupe B
        [5, 5],  # Point ambigu
    ]
    
    n = len(points)
    
    # Étape 1: Calculer les distances (manuellement)
    distances = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                distances[i, j] = np.sqrt((points[i][0]-points[j][0])**2 + 
                                         (points[i][1]-points[j][1])**2)
    
    # Étape 2: Trouver les k plus proches voisins (k=2)
    k = 2
    neighbors = []
    for i in range(n):
        # Indices des k+1 plus proches (inclut soi-même)
        idx = np.argsort(distances[i])[:k+1]
        neighbors.append(set(idx))
    
    # Étape 3: Calculer les similarités SNN
    print(f"\nVoisins (k={k}):")
    for i, neigh in enumerate(neighbors):
        print(f"  Point {i}{points[i]} : voisins {list(neigh)}")
    
    print(f"\nSimilarités SNN (nombre de voisins communs):")
    for i in range(n):
        for j in range(i+1, n):
            common = len(neighbors[i].intersection(neighbors[j]))
            print(f"  Similarité({i},{j}) = {common}")
    
    # Étape 4: Clustering simple (seuil = 2)
    threshold = 2
    clusters = [-1] * n  # -1 = non assigné
    current_cluster = 0
    
    for i in range(n):
        if clusters[i] == -1:
            # Trouver tous les points connectés à i
            cluster_points = [i]
            to_check = [i]
            
            while to_check:
                current = to_check.pop()
                
                for j in range(n):
                    if j != current and clusters[j] == -1:
                        common = len(neighbors[current].intersection(neighbors[j]))
                        if common >= threshold:
                            cluster_points.append(j)
                            to_check.append(j)
                            clusters[j] = current_cluster
            
            # Vérifier la taille du cluster
            if len(cluster_points) > 1:
                for p in cluster_points:
                    clusters[p] = current_cluster
                current_cluster += 1
    
    print(f"\nClusters trouvés:")
    for i, cluster in enumerate(clusters):
        if cluster == -1:
            print(f"  Point {i}{points[i]} → Bruit")
        else:
            print(f"  Point {i}{points[i]} → Cluster {cluster}")
    
    return clusters

# Fonction principale
def main():
    """Exécute tous les exemples SNN."""
    print("=" * 60)
    print("CLUSTERING PAR SHARED NEAREST NEIGHBOR (SNN)")
    print("=" * 60)
    
    # Installation nécessaire
    print("\nInstallation requise:")
    print("pip install numpy matplotlib scikit-learn networkx")
    
    try:
        import sklearn
        import networkx as nx
        print("✓ Toutes les dépendances sont installées")
    except ImportError as e:
        print(f"\n⚠ Dépendance manquante: {e}")
        print("Installez avec: pip install numpy matplotlib scikit-learn networkx")
    
    # Exemple 1 : Simple
    print("\n1. Exemple simple avec 6 points")
    snn1, labels1 = example_snn_simple()
    
    # Exemple 2 : Densités variables
    print("\n2. Clusters de densités différentes")
    snn2, labels2 = example_variable_density()
    
    # Exemple 3 : Comparaison avec DBSCAN
    print("\n3. Comparaison SNN vs DBSCAN")
    snn3, dbscan3, labels_snn3, labels_db3 = compare_snn_dbscan()
    
    # Exemple 4 : Données textuelles
    print("\n4. Application aux données textuelles")
    snn4, labels4, docs4 = snn_text_example()
    
    # Exemple 5 : Version ultra-simplifiée
    print("\n5. Version ultra-simplifiée (compréhension)")
    clusters5 = ultra_simple_snn()
    
    print("\n" + "=" * 60)
    print("DÉMONSTRATION TERMINÉE")
    print("=" * 60)

if __name__ == "__main__":
    main()
