import metis
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from scipy.spatial.distance import pdist, squareform

class MetisPartitioner:
    """
    Classe utilitaire pour appliquer METIS à différents problèmes de data mining
    """
    
    def __init__(self, n_parts=4):
        self.n_parts = n_parts
        self.partitions = None
        self.edgecut = None
        
    def partition_graph(self, adjacency_list, node_weights=None, edge_weights=None):
        """
        Partitionne un graphe représenté par une liste d'adjacence
        
        Parameters:
        -----------
        adjacency_list : list de list, liste d'adjacence du graphe
        node_weights : list, poids des nœuds (optionnel)
        edge_weights : list de list, poids des arêtes (optionnel)
        """
        # Créer le graphe au format METIS
        graph = metis.adjlist_to_metis(adjacency_list, 
                                       vweights=node_weights, 
                                       eweights=edge_weights)
        
        # Partitionnement
        (self.edgecut, self.partitions) = metis.part_graph(graph, self.n_parts)
        
        return self.partitions, self.edgecut
    
    def partition_from_matrix(self, similarity_matrix, threshold=0.5):
        """
        Partitionne des données à partir d'une matrice de similarité
        
        Parameters:
        -----------
        similarity_matrix : array 2D, matrice de similarité
        threshold : float, seuil pour créer des arêtes
        """
        n = similarity_matrix.shape[0]
        
        # Créer le graphe à partir de la matrice de similarité
        adjacency_list = []
        for i in range(n):
            neighbors = []
            for j in range(n):
                if i != j and similarity_matrix[i, j] > threshold:
                    neighbors.append(j)
            adjacency_list.append(neighbors)
        
        return self.partition_graph(adjacency_list)
    
    def partition_geometric_data(self, X, k_neighbors=10):
        """
        Partitionne des données géométriques en créant un graphe k-NN
        
        Parameters:
        -----------
        X : array 2D, données (n_samples, n_features)
        k_neighbors : int, nombre de voisins pour k-NN
        """
        from sklearn.neighbors import kneighbors_graph
        
        # Créer le graphe k-NN
        A = kneighbors_graph(X, k_neighbors, mode='connectivity', include_self=False)
        
        # Convertir en liste d'adjacence
        adjacency_list = [list(A[i].nonzero()[1]) for i in range(X.shape[0])]
        
        return self.partition_graph(adjacency_list)

# Exemple 1 : Partitionnement d'un graphe aléatoire
def example_random_graph():
    """Exemple avec un graphe aléatoire"""
    print("=== Exemple 1 : Graphe aléatoire ===")
    
    # Créer un graphe aléatoire
    n_nodes = 100
    G = nx.erdos_renyi_graph(n_nodes, 0.1)
    
    # Convertir en liste d'adjacence
    adjacency_list = [list(G.neighbors(i)) for i in range(n_nodes)]
    
    # Partitionner avec METIS
    partitioner = MetisPartitioner(n_parts=4)
    partitions, edgecut = partitioner.partition_graph(adjacency_list)
    
    print(f"Nombre de nœuds : {n_nodes}")
    print(f"Nombre d'arêtes coupées : {edgecut}")
    print(f"Taille des partitions : {np.bincount(partitions)}")
    
    # Visualisation
    pos = nx.spring_layout(G)
    colors = ['red', 'blue', 'green', 'orange']
    
    plt.figure(figsize=(10, 5))
    for i in range(4):
        nodes = [j for j, p in enumerate(partitions) if p == i]
        nx.draw_networkx_nodes(G, pos, nodelist=nodes, 
                              node_color=colors[i], node_size=50, label=f'Partition {i}')
    
    nx.draw_networkx_edges(G, pos, alpha=0.3)
    plt.title(f"Partitionnement METIS (Edge cut: {edgecut})")
    plt.legend()
    plt.show()
    
    return partitions

# Exemple 2 : Clustering de données avec graphe k-NN
def example_data_clustering():
    """Exemple de clustering de données avec METIS via graphe k-NN"""
    print("\n=== Exemple 2 : Clustering de données ===")
    
    # Générer des données
    X, y_true = make_blobs(n_samples=200, centers=4, cluster_std=0.6, random_state=42)
    
    # Partitionner avec METIS via graphe k-NN
    partitioner = MetisPartitioner(n_parts=4)
    partitions, edgecut = partitioner.partition_geometric_data(X, k_neighbors=10)
    
    # Évaluation
    from sklearn.metrics import adjusted_rand_score
    ari = adjusted_rand_score(y_true, partitions)
    
    print(f"Adjusted Rand Index : {ari:.3f}")
    print(f"Edge cut : {edgecut}")
    
    # Visualisation
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Vrais clusters
    scatter1 = axes[0].scatter(X[:, 0], X[:, 1], c=y_true, cmap='tab10', s=30)
    axes[0].set_title('Vrais clusters')
    axes[0].set_xlabel('Feature 1')
    axes[0].set_ylabel('Feature 2')
    
    # Partitionnement METIS
    scatter2 = axes[1].scatter(X[:, 0], X[:, 1], c=partitions, cmap='tab10', s=30)
    axes[1].set_title(f'Partitionnement METIS (ARI={ari:.3f})')
    axes[1].set_xlabel('Feature 1')
    axes[1].set_ylabel('Feature 2')
    
    plt.tight_layout()
    plt.show()
    
    return partitions

# Exemple 3 : Partitionnement pour calcul parallèle
def example_parallel_computing():
    """Exemple de partitionnement pour calcul parallèle"""
    print("\n=== Exemple 3 : Partitionnement pour calcul parallèle ===")
    
    # Simuler un problème avec 1000 éléments interconnectés
    n_elements = 1000
    
    # Créer un graphe avec structure de bloc (simulant des dépendances)
    np.random.seed(42)
    
    # Matrice d'adjacence (probabilité de connexion plus forte dans les blocs)
    adjacency = np.zeros((n_elements, n_elements))
    
    # Créer 4 blocs naturels
    block_size = n_elements // 4
    for block in range(4):
        start = block * block_size
        end = start + block_size
        # Forte connectivité à l'intérieur du bloc
        for i in range(start, end):
            for j in range(i+1, end):
                if np.random.random() < 0.3:
                    adjacency[i, j] = 1
                    adjacency[j, i] = 1
    
    # Quelques connexions entre blocs
    n_between = 100
    for _ in range(n_between):
        i = np.random.randint(0, n_elements)
        j = np.random.randint(0, n_elements)
        if i // block_size != j // block_size:  # Différents blocs
            adjacency[i, j] = 1
            adjacency[j, i] = 1
    
    # Convertir en liste d'adjacence
    adjacency_list = [list(np.where(row == 1)[0]) for row in adjacency]
    
    # Partitionner pour 4 processeurs
    partitioner = MetisPartitioner(n_parts=4)
    partitions, edgecut = partitioner.partition_graph(adjacency_list)
    
    # Analyser les résultats
    partition_sizes = np.bincount(partitions)
    print(f"Taille des partitions : {partition_sizes}")
    print(f"Déséquilibre : {np.max(partition_sizes) / np.min(partition_sizes):.2f}x")
    print(f"Arêtes coupées : {edgecut}")
    print(f"Pourcentage d'arêtes coupées : {edgecut / np.sum(adjacency) * 100:.1f}%")
    
    # Visualiser la matrice de connectivité après partitionnement
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Matrice d'adjacence originale
    axes[0].imshow(adjacency, cmap='Greys', aspect='auto')
    axes[0].set_title('Matrice d\'adjacence originale')
    axes[0].set_xlabel('Élément')
    axes[0].set_ylabel('Élément')
    
    # Réorganiser selon les partitions
    sorted_indices = np.argsort(partitions)
    sorted_adjacency = adjacency[sorted_indices][:, sorted_indices]
    
    axes[1].imshow(sorted_adjacency, cmap='Greys', aspect='auto')
    axes[1].set_title('Matrice réorganisée par partitions METIS')
    axes[1].set_xlabel('Élément (trié par partition)')
    axes[1].set_ylabel('Élément (trié par partition)')
    
    # Ajouter des lignes de séparation entre partitions
    cum_sizes = np.cumsum(partition_sizes)
    for size in cum_sizes[:-1]:
        axes[1].axhline(y=size, color='red', linestyle='--', linewidth=1)
        axes[1].axvline(x=size, color='red', linestyle='--', linewidth=1)
    
    plt.tight_layout()
    plt.show()
    
    return partitions

# Exemple 4 : Détection de communautés dans un réseau social
def example_community_detection():
    """Exemple de détection de communautés avec METIS"""
    import networkx as nx
    
    print("\n=== Exemple 4 : Détection de communautés ===")
    
    # Créer un graphe avec structure communautaire
    n_communities = 4
    community_size = 25
    G = nx.planted_partition_graph(n_communities, community_size, 0.8, 0.1, seed=42)
    
    # Convertir en liste d'adjacence
    adjacency_list = [list(G.neighbors(i)) for i in range(len(G))]
    
    # Détecter les communautés avec METIS
    partitioner = MetisPartitioner(n_parts=n_communities)
    partitions, edgecut = partitioner.partition_graph(adjacency_list)
    
    # Évaluation (comparaison avec la vérité terrain)
    true_partitions = [i // community_size for i in range(len(G))]
    
    from sklearn.metrics import normalized_mutual_info_score
    nmi = normalized_mutual_info_score(true_partitions, partitions)
    
    print(f"Normalized Mutual Information : {nmi:.3f}")
    print(f"Arêtes coupées : {edgecut}")
    
    # Visualisation
    pos = nx.spring_layout(G, seed=42)
    
    plt.figure(figsize=(10, 5))
    
    # Vraies communautés
    plt.subplot(1, 2, 1)
    colors_true = [true_partitions[i] for i in range(len(G))]
    nx.draw(G, pos, node_color=colors_true, cmap='tab10', 
            node_size=50, with_labels=False)
    plt.title('Vraies communautés')
    
    # Communautés détectées par METIS
    plt.subplot(1, 2, 2)
    nx.draw(G, pos, node_color=partitions, cmap='tab10', 
            node_size=50, with_labels=False)
    plt.title(f'Communautés METIS (NMI={nmi:.3f})')
    
    plt.tight_layout()
    plt.show()
    
    return partitions

if __name__ == "__main__":
    # Exécuter les exemples
    partitions1 = example_random_graph()
    partitions2 = example_data_clustering()
    partitions3 = example_parallel_computing()
    partitions4 = example_community_detection()
    
"""
pour compiler, faire
sudo apt update
sudo apt install libmetis-dev

export METIS_DLL="/usr/lib/x86_64-linux-gnu/libmetis.so"
python3 METIS.py

"""
