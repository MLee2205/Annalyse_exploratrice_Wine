import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

class SimpleRanker:
    """Classificateur simple pour ranking."""
    
    def __init__(self, method='weighted', weights=None):
        """
        Initialise le ranker.
        
        Parameters:
        -----------
        method : str, méthode de ranking
                 'weighted', 'borda', 'pagerank'
        weights : dict, poids pour méthode weighted
        """
        self.method = method
        
        if weights is None:
            self.weights = {
                'rating': 0.4,
                'reviews': 0.2,
                'relevance': 0.3,
                'conversion': 0.1
            }
        else:
            self.weights = weights
        
    def fit(self, X, feature_names=None):
        """Prépare les données pour le ranking."""
        self.X = np.array(X)
        self.n_items = len(X)
        
        if feature_names is None:
            self.feature_names = [f'Feature_{i}' for i in range(X.shape[1])]
        else:
            self.feature_names = feature_names
        
        return self
    
    def rank_weighted(self):
        """Ranking par score pondéré."""
        scores = np.zeros(self.n_items)
        
        # Normaliser les features
        X_norm = MinMaxScaler().fit_transform(self.X)
        
        # Appliquer les poids
        if len(self.weights) == self.X.shape[1]:
            weights_array = np.array(list(self.weights.values()))
            scores = np.dot(X_norm, weights_array)
        else:
            # Par défaut: poids égaux
            scores = np.mean(X_norm, axis=1)
        
        return self._get_ranking_from_scores(scores)
    
    def rank_borda(self):
        """Ranking par méthode de Borda."""
        n_items = self.n_items
        n_features = self.X.shape[1]
        
        # Points Borda: 1er = n_items-1 pts, dernier = 0 pt
        borda_points = np.zeros(n_items)
        
        for feature_idx in range(n_features):
            # Classer par cette feature
            feature_values = self.X[:, feature_idx]
            ranks = np.argsort(feature_values)[::-1]  # Descendant
            
            # Attribuer les points
            for position, item_idx in enumerate(ranks):
                points = n_items - position - 1
                borda_points[item_idx] += points
        
        return self._get_ranking_from_scores(borda_points)
    
    def rank_pagerank(self, alpha=0.85, max_iter=100, tol=1e-6):
        """Ranking par PageRank simplifié."""
        n = self.n_items
        
        # Créer une matrice de similarité basée sur les features
        similarity_matrix = self._create_similarity_matrix()
        
        # Normaliser la matrice (chaque ligne somme à 1)
        row_sums = similarity_matrix.sum(axis=1)
        row_sums[row_sums == 0] = 1  # Éviter division par 0
        M = similarity_matrix / row_sums[:, np.newaxis]
        
        # Initialisation
        pr = np.ones(n) / n
        
        # Itérations PageRank
        for _ in range(max_iter):
            new_pr = (1 - alpha) / n + alpha * M.T.dot(pr)
            
            # Vérifier la convergence
            if np.abs(new_pr - pr).sum() < tol:
                break
            pr = new_pr
        
        return self._get_ranking_from_scores(pr)
    
    def _create_similarity_matrix(self):
        """Crée une matrice de similarité entre items."""
        n = self.n_items
        similarity = np.zeros((n, n))
        
        # Similarité cosinus
        for i in range(n):
            for j in range(n):
                if i != j:
                    # Similarité basée sur les features
                    dot_product = np.dot(self.X[i], self.X[j])
                    norm_i = np.linalg.norm(self.X[i])
                    norm_j = np.linalg.norm(self.X[j])
                    
                    if norm_i > 0 and norm_j > 0:
                        similarity[i, j] = dot_product / (norm_i * norm_j)
        
        return similarity
    
    def _get_ranking_from_scores(self, scores):
        """Convertit des scores en classement."""
        # argsort retourne les indices triés (ascendant)
        # On inverse pour avoir descendant
        ranked_indices = np.argsort(scores)[::-1]
        
        ranking = np.zeros(self.n_items, dtype=int)
        for position, idx in enumerate(ranked_indices):
            ranking[idx] = position + 1
        
        return ranking, scores
    
    def rank(self):
        """Exécute le ranking selon la méthode choisie."""
        if self.method == 'weighted':
            ranking, scores = self.rank_weighted()
        elif self.method == 'borda':
            ranking, scores = self.rank_borda()
        elif self.method == 'pagerank':
            ranking, scores = self.rank_pagerank()
        else:
            raise ValueError(f"Méthode non reconnue: {self.method}")
        
        return ranking, scores

# Exemple avec nos données manuelles
def example_manual_ranking():
    """Exemple de ranking avec nos données produits."""
    
    print("=" * 60)
    print("EXEMPLE DE RANKING DE PRODUITS")
    print("=" * 60)
    
    # Nos données
    data = np.array([
        [4.8, 1200, 0.9, 5.2],  # A
        [4.5, 50,   0.7, 3.1],  # B
        [4.9, 20,   0.8, 2.8],  # C
        [4.2, 300,  0.6, 4.5],  # D
        [3.9, 500,  0.5, 6.0]   # E
    ])
    
    product_names = ['A', 'B', 'C', 'D', 'E']
    feature_names = ['Note', 'Avis', 'Pertinence', 'Conversion']
    
    # Créer un DataFrame pour affichage
    df = pd.DataFrame(data, index=product_names, columns=feature_names)
    print("\nDonnées des produits:")
    print(df)
    
    # Tester différentes méthodes
    methods = ['weighted', 'borda', 'pagerank']
    all_rankings = {}
    
    for method in methods:
        print(f"\n{'='*40}")
        print(f"MÉTHODE: {method.upper()}")
        print('='*40)
        
        ranker = SimpleRanker(method=method)
        ranker.fit(data, feature_names=feature_names)
        ranking, scores = ranker.rank()
        
        all_rankings[method] = ranking
        
        # Afficher les résultats
        results = pd.DataFrame({
            'Produit': product_names,
            'Score': scores.round(3),
            'Rang': ranking
        })
        
        results = results.sort_values('Rang')
        print(results.to_string(index=False))
    
    # Comparaison des méthodes
    print(f"\n{'='*60}")
    print("COMPARAISON DES MÉTHODES")
    print('='*60)
    
    comparison = pd.DataFrame(index=product_names)
    for method in methods:
        comparison[method] = all_rankings[method]
    
    # Rang moyen
    comparison['Rang moyen'] = comparison.mean(axis=1).round(2)
    comparison['Rang final'] = comparison['Rang moyen'].rank(method='min').astype(int)
    
    print("\nRangs par méthode:")
    print(comparison.sort_values('Rang final'))
    
    return all_rankings, comparison

# Version 2 : Learning to Rank (LTR) simplifié
class LearningToRankSimple:
    """Learning to Rank simplifié avec régression."""
    
    def __init__(self):
        self.model = None
        
    def fit(self, X, y, groups):
        """
        Entraîne un modèle de ranking.
        
        Parameters:
        -----------
        X : features des items
        y : scores de pertinence (plus élevé = plus pertinent)
        groups : groupe d'appartenance (ex: query_id)
        """
        from sklearn.linear_model import LinearRegression
        
        # Pour simplifier, on utilise une régression linéaire
        self.model = LinearRegression()
        self.model.fit(X, y)
        
        return self
    
    def predict_rank(self, X, groups):
        """Prédit le ranking pour de nouveaux items."""
        scores = self.model.predict(X)
        
        # Grouper par query et ranker dans chaque groupe
        unique_groups = np.unique(groups)
        rankings = np.zeros(len(X), dtype=int)
        
        for group in unique_groups:
            group_indices = np.where(groups == group)[0]
            group_scores = scores[group_indices]
            
            # Ranker dans le groupe
            group_ranks = np.argsort(group_scores)[::-1]  # Descendant
            for pos, idx in enumerate(group_ranks):
                original_idx = group_indices[idx]
                rankings[original_idx] = pos + 1
        
        return rankings, scores

# Exemple 2 : Ranking de résultats de recherche
def search_ranking_example():
    """Exemple de ranking pour moteur de recherche."""
    
    print("\n" + "=" * 60)
    print("RANKING DE RÉSULTATS DE RECHERCHE")
    print("=" * 60)
    
    # Simuler des données de recherche
    np.random.seed(42)
    
    # 3 requêtes, 5 résultats par requête
    n_queries = 3
    n_results_per_query = 5
    
    # Features pour chaque résultat
    # [tf-idf, pagerank, freshness, click_through_rate]
    X = []
    y = []  # Pertinence réelle (0-4)
    groups = []  # Query ID
    
    for query_id in range(n_queries):
        for result_id in range(n_results_per_query):
            # Features simulées
            tfidf = np.random.uniform(0.1, 1.0)
            pagerank = np.random.uniform(0.01, 0.1)
            freshness = np.random.uniform(0, 1)  # 1 = récent
            ctr = np.random.uniform(0.01, 0.1)
            
            X.append([tfidf, pagerank, freshness, ctr])
            groups.append(query_id)
            
            # Pertinence simulée (basée sur les features)
            relevance = (0.5 * tfidf + 0.3 * pagerank + 0.1 * freshness + 0.1 * ctr)
            relevance_score = int(relevance * 4)  # Convertir en 0-4
            y.append(relevance_score)
    
    X = np.array(X)
    y = np.array(y)
    groups = np.array(groups)
    
    print(f"Données générées:")
    print(f"  • Requêtes: {n_queries}")
    print(f"  • Résultats totaux: {len(X)}")
    print(f"  • Features: TF-IDF, PageRank, Fraîcheur, CTR")
    print(f"  • Pertinence: 0 (faible) à 4 (forte)")
    
    # Ranking avec différentes méthodes
    print(f"\n{'='*50}")
    print("RANKING PAR MÉTHODE WEIGHTED")
    print('='*50)
    
    ranker_weighted = SimpleRanker(method='weighted')
    ranker_weighted.fit(X)
    ranking_w, scores_w = ranker_weighted.rank_weighted()
    
    # Afficher par requête
    for query_id in range(n_queries):
        query_indices = np.where(groups == query_id)[0]
        
        print(f"\nRequête {query_id + 1}:")
        print("-" * 40)
        
        results = []
        for idx in query_indices:
            results.append({
                'Résultat': f"R{idx%n_results_per_query + 1}",
                'TF-IDF': X[idx, 0],
                'PageRank': X[idx, 1],
                'Fraîcheur': X[idx, 2],
                'CTR': X[idx, 3],
                'Pertinence réelle': y[idx],
                'Score': scores_w[idx],
                'Rang': ranking_w[idx]
            })
        
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values('Rang')
        print(results_df.to_string(index=False))
    
    # Évaluation avec NDCG
    print(f"\n{'='*50}")
    print("ÉVALUATION AVEC NDCG")
    print('='*50)
    
    for query_id in range(n_queries):
        query_indices = np.where(groups == query_id)[0]
        
        # Pertinence réelle
        true_relevance = y[query_indices]
        
        # Ranking prédit (par score)
        query_scores = scores_w[query_indices]
        predicted_order = np.argsort(query_scores)[::-1]
        predicted_relevance = true_relevance[predicted_order]
        
        # Calculer DCG et IDCG
        dcg = 0
        idcg = 0
        
        # Ordre idéal (par pertinence décroissante)
        ideal_order = np.argsort(true_relevance)[::-1]
        ideal_relevance = true_relevance[ideal_order]
        
        for i in range(len(predicted_order)):
            # DCG@k (k = tous les résultats)
            rel = predicted_relevance[i]
            dcg += rel / np.log2(i + 2)  # i+2 car log2(1) = 0
            
            # IDCG@k
            ideal_rel = ideal_relevance[i]
            idcg += ideal_rel / np.log2(i + 2)
        
        ndcg = dcg / idcg if idcg > 0 else 0
        
        print(f"Requête {query_id + 1}: NDCG = {ndcg:.3f}")
        print(f"  Pertinence réelle triée: {true_relevance[predicted_order]}")
        print(f"  Ordre idéal: {ideal_relevance}")
    
    return X, y, groups, ranking_w, scores_w

# Version 3 : RankNet simplifié (pairwise ranking)
class RankNetSimple:
    """Implémentation simplifiée de RankNet."""
    
    def __init__(self, learning_rate=0.01, n_epochs=100):
        self.lr = learning_rate
        self.n_epochs = n_epochs
        self.weights = None
        
    def fit(self, X, pairs, labels):
        """
        Entraîne RankNet sur des paires.
        
        Parameters:
        -----------
        X : features des items
        pairs : liste de tuples (i, j) où i est préféré à j
        labels : labels des paires (1 si i > j, 0 sinon)
        """
        n_features = X.shape[1]
        self.weights = np.random.randn(n_features) * 0.01
        
        # Entraînement
        for epoch in range(self.n_epochs):
            total_loss = 0
            
            for (i, j), label in zip(pairs, labels):
                # Features des deux items
                x_i = X[i]
                x_j = X[j]
                
                # Scores prédits
                s_i = np.dot(self.weights, x_i)
                s_j = np.dot(self.weights, x_j)
                
                # Différence de scores
                s_diff = s_i - s_j
                
                # Probabilité que i soit meilleur que j
                p_ij = 1 / (1 + np.exp(-s_diff))
                
                # Loss (cross-entropy)
                loss = -label * np.log(p_ij) - (1 - label) * np.log(1 - p_ij)
                total_loss += loss
                
                # Gradient
                grad = (p_ij - label) * (x_i - x_j)
                
                # Mise à jour des poids
                self.weights -= self.lr * grad
            
            if epoch % 20 == 0:
                print(f"Epoch {epoch}: loss = {total_loss/len(pairs):.4f}")
        
        return self
    
    def predict_scores(self, X):
        """Prédit les scores pour des items."""
        return np.dot(X, self.weights)
    
    def predict_ranking(self, X):
        """Prédit le ranking."""
        scores = self.predict_scores(X)
        ranking = np.argsort(scores)[::-1]  # Indices triés descendant
        return ranking, scores

# Exemple 3 : Ranking par paires (pairwise)
def pairwise_ranking_example():
    """Exemple de ranking avec apprentissage par paires."""
    
    print("\n" + "=" * 60)
    print("RANKING PAR PAIRES (PAIRWISE)")
    print("=" * 60)
    
    # Données de films
    movies = [
        "Inception",
        "The Dark Knight",
        "Interstellar",
        "The Shawshank Redemption",
        "Pulp Fiction",
        "Fight Club"
    ]
    
    # Features: [note_imdb, nb_votes, durée, année]
    X = np.array([
        [8.8, 2300000, 148, 2010],  # Inception
        [9.0, 2700000, 152, 2008],  # TDK
        [8.6, 1800000, 169, 2014],  # Interstellar
        [9.3, 2700000, 142, 1994],  # Shawshank
        [8.9, 2000000, 154, 1994],  # Pulp Fiction
        [8.8, 2200000, 139, 1999]   # Fight Club
    ])
    
    # Créer des paires de préférences (basées sur les notes IMDB)
    pairs = []
    labels = []
    
    for i in range(len(movies)):
        for j in range(i+1, len(movies)):
            # Si i a une note plus élevée que j
            if X[i, 0] > X[j, 0]:
                pairs.append((i, j))
                labels.append(1)  # i est préféré à j
            elif X[i, 0] < X[j, 0]:
                pairs.append((j, i))
                labels.append(1)  # j est préféré à i
    
    print(f"Données de {len(movies)} films")
    print(f"Paires d'entraînement: {len(pairs)}")
    
    # Entraîner RankNet
    ranknet = RankNetSimple(learning_rate=0.01, n_epochs=100)
    ranknet.fit(X, pairs, labels)
    
    # Prédire le ranking
    ranking, scores = ranknet.predict_ranking(X)
    
    print(f"\n{'='*50}")
    print("CLASSEMENT DES FILMS")
    print('='*50)
    
    results = []
    for position, idx in enumerate(ranking):
        results.append({
            'Rang': position + 1,
            'Film': movies[idx],
            'Note IMDB': X[idx, 0],
            'Votes (M)': X[idx, 1] / 1000000,
            'Score prédit': scores[idx]
        })
    
    results_df = pd.DataFrame(results)
    print(results_df.to_string(index=False))
    
    # Comparer avec le ranking par note IMDB
    print(f"\n{'='*50}")
    print("COMPARAISON AVEC NOTES IMDB")
    print('='*50)
    
    # Ranking par note IMDB
    imdb_ranking = np.argsort(X[:, 0])[::-1]
    
    comparison = pd.DataFrame({
        'Film': movies,
        'Note IMDB': X[:, 0],
        'Rang par RankNet': [np.where(ranking == i)[0][0] + 1 for i in range(len(movies))],
        'Rang par Note': [np.where(imdb_ranking == i)[0][0] + 1 for i in range(len(movies))]
    })
    
    comparison['Différence'] = comparison['Rang par RankNet'] - comparison['Rang par Note']
    comparison = comparison.sort_values('Rang par RankNet')
    print(comparison.to_string(index=False))
    
    return ranknet, X, movies, ranking, scores

# Exemple 4 : Ranking avec feedback utilisateur
def user_feedback_ranking():
    """Exemple de ranking avec feedback utilisateur."""
    
    print("\n" + "=" * 60)
    print("RANKING AVEC FEEDBACK UTILISATEUR")
    print("=" * 60)
    
    # Simuler des clics utilisateur
    np.random.seed(123)
    
    # 10 articles, 4 features
    n_articles = 10
    X = np.random.randn(n_articles, 4)
    
    # Pertinence réelle (cachée)
    true_relevance = np.dot(X, [0.5, 0.3, -0.2, 0.4]) + np.random.normal(0, 0.1, n_articles)
    
    # Simuler des sessions utilisateur
    n_sessions = 50
    clicks = []
    
    print(f"Simulation de {n_sessions} sessions utilisateur...")
    
    for session in range(n_sessions):
        # L'utilisateur voit 3 articles aléatoires
        shown_articles = np.random.choice(n_articles, 3, replace=False)
        
        # Probabilité de clic basée sur la pertinence
        for article_idx in shown_articles:
            click_prob = 1 / (1 + np.exp(-true_relevance[article_idx]))
            clicked = np.random.binomial(1, click_prob)
            
            if clicked:
                clicks.append({
                    'session': session,
                    'article': article_idx,
                    'position': np.where(shown_articles == article_idx)[0][0],
                    'clicked': 1
                })
            else:
                clicks.append({
                    'session': session,
                    'article': article_idx,
                    'position': np.where(shown_articles == article_idx)[0][0],
                    'clicked': 0
                })
    
    clicks_df = pd.DataFrame(clicks)
    
    print(f"\nStatistiques des clics:")
    print(f"  • Total impressions: {len(clicks_df)}")
    print(f"  • Clics: {clicks_df['clicked'].sum()}")
    print(f"  • Taux de clic: {clicks_df['clicked'].mean():.1%}")
    
    # Apprendre un modèle de ranking basé sur les clics
    print(f"\n{'='*50}")
    print("APPRENTISSAGE DU RANKING")
    print('='*50)
    
    # Créer des paires (article cliqué > article non cliqué dans même session)
    pairs = []
    
    for session_id in clicks_df['session'].unique():
        session_data = clicks_df[clicks_df['session'] == session_id]
        
        # Articles cliqués dans cette session
        clicked_articles = session_data[session_data['clicked'] == 1]['article'].values
        
        # Articles non cliqués
        not_clicked_articles = session_data[session_data['clicked'] == 0]['article'].values
        
        # Créer des paires
        for clicked in clicked_articles:
            for not_clicked in not_clicked_articles:
                pairs.append((clicked, not_clicked))
    
    print(f"Pairs générées: {len(pairs)}")
    
    if len(pairs) > 0:
        # Entraîner un modèle pairwise
        labels = [1] * len(pairs)  # Toujours clicked > not clicked
        
        ranknet = RankNetSimple(learning_rate=0.01, n_epochs=50)
        ranknet.fit(X, pairs, labels)
        
        # Ranking final
        ranking, scores = ranknet.predict_ranking(X)
        
        print(f"\n{'='*50}")
        print("CLASSEMENT FINAL DES ARTICLES")
        print('='*50)
        
        results = []
        for position, idx in enumerate(ranking):
            click_count = clicks_df[clicks_df['article'] == idx]['clicked'].sum()
            impressions = len(clicks_df[clicks_df['article'] == idx])
            ctr = click_count / impressions if impressions > 0 else 0
            
            results.append({
                'Rang': position + 1,
                'Article': f"Article_{idx}",
                'Score prédit': scores[idx],
                'Clics': click_count,
                'Impressions': impressions,
                'CTR': f"{ctr:.1%}"
            })
        
        results_df = pd.DataFrame(results)
        print(results_df.to_string(index=False))
        
        # Vérifier la qualité
        print(f"\n{'='*50}")
        print("VÉRIFICATION DE LA QUALITÉ")
        print('='*50)
        
        # Correlation avec le CTR réel
        article_stats = clicks_df.groupby('article').agg({
            'clicked': ['sum', 'count']
        }).reset_index()
        article_stats.columns = ['article', 'clicks', 'impressions']
        article_stats['ctr'] = article_stats['clicks'] / article_stats['impressions']
        
        # Fusionner avec les scores prédits
        article_stats['score'] = scores[article_stats['article'].values]
        
        correlation = np.corrcoef(article_stats['score'], article_stats['ctr'])[0, 1]
        print(f"Corrélation score-CTR: {correlation:.3f}")
        
        return ranknet, X, clicks_df, ranking, scores
    
    return None

# Fonction principale
def main():
    """Exécute tous les exemples de ranking."""
    print("=" * 60)
    print("RANKING - CLASSEMENT D'ITEMS")
    print("=" * 60)
    
    # Exemple 1 : Ranking basique
    print("\n1. RANKING BASIQUE DE PRODUITS")
    all_rankings, comparison = example_manual_ranking()
    
    # Exemple 2 : Ranking de recherche
    print("\n2. RANKING DE RECHERCHE")
    X_search, y_search, groups, ranking_w, scores_w = search_ranking_example()
    
    # Exemple 3 : Ranking par paires
    print("\n3. RANKING PAR PAIRES (FILMS)")
    ranknet, X_movies, movies, movie_ranking, movie_scores = pairwise_ranking_example()
    
    # Exemple 4 : Ranking avec feedback
    print("\n4. RANKING AVEC FEEDBACK UTILISATEUR")
    result = user_feedback_ranking()
    
    print("\n" + "=" * 60)
    print("CONCLUSION SUR LE RANKING")
    print("=" * 60)
    print("""
    Le ranking transforme des scores en positions relatives.
    
    Méthodes principales:
    1. Score-based : Score unique → tri
    2. Pairwise : Apprentissage par comparaisons deux à deux
    3. Listwise : Optimisation directe du classement complet
    
    Mesures d'évaluation:
    • NDCG : Prise en compte de la position
    • MAP : Pour la recherche d'information
    • MRR : Pour la première réponse correcte
    
    Applications:
    • Moteurs de recherche (Google, Bing)
    • Recommandation (Netflix, Amazon)
    • Publicité en ligne (classement des annonces)
    • Réseaux sociaux (fil d'actualité)
    
    Challenges:
    • Cold start (nouveaux items/utilisateurs)
    • Biais de position
    • Évolutivité (millions d'items)
    • Personnalisation
    """)

if __name__ == "__main__":
    main()
