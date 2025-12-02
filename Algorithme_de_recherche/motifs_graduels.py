import numpy as np
import pandas as pd
from itertools import combinations

class GradualPatternMiner:
    """Mineur de motifs graduels simple."""
    
    def __init__(self, min_support=0.3):
        self.min_support = min_support
        self.patterns = []
        
    def fit(self, data):
        """
        Trouve les motifs graduels dans les données.
        
        Parameters:
        -----------
        data : DataFrame ou array 2D, données numériques
        """
        if isinstance(data, pd.DataFrame):
            data = data.values
        
        n_objects, n_attributes = data.shape
        total_pairs = n_objects * (n_objects - 1) // 2
        
        print(f"Données: {n_objects} objets, {n_attributes} attributs")
        print(f"Nombre total de paires: {total_pairs}")
        print("-" * 50)
        
        # Pour chaque paire d'attributs
        for i in range(n_attributes):
            for j in range(i+1, n_attributes):
                # Tester les 4 combinaisons de variations
                patterns_to_test = [
                    (f"A{i}+", f"A{j}+"),   # Les deux augmentent
                    (f"A{i}+", f"A{j}-"),   # i augmente, j diminue
                    (f"A{i}-", f"A{j}+"),   # i diminue, j augmente
                    (f"A{i}-", f"A{j}-")    # Les deux diminuent
                ]
                
                variations = [
                    (1, 1),   # (+, +)
                    (1, -1),  # (+, -)
                    (-1, 1),  # (-, +)
                    (-1, -1)  # (-, -)
                ]
                
                for pattern, (var_i, var_j) in zip(patterns_to_test, variations):
                    support = self._compute_support(data, i, j, var_i, var_j)
                    
                    if support >= self.min_support:
                        self.patterns.append({
                            'pattern': pattern,
                            'support': support,
                            'attributs': (i, j),
                            'variations': (var_i, var_j)
                        })
                        
                        print(f"Motif: {pattern[0]} et {pattern[1]}")
                        print(f"  Support: {support:.1%}")
                        print(f"  Paires valides: {int(support * total_pairs)}/{total_pairs}")
                        
        return self
    
    def _compute_support(self, data, attr_i, attr_j, var_i, var_j):
        """Calcule le support d'un motif à 2 attributs."""
        n = len(data)
        valid_pairs = 0
        
        for k in range(n):
            for l in range(k+1, n):
                # Vérifier la variation pour le premier attribut
                if var_i == 1:  # Augmentation
                    valid_i = data[k, attr_i] < data[l, attr_i]
                else:  # Diminution
                    valid_i = data[k, attr_i] > data[l, attr_i]
                
                # Vérifier la variation pour le second attribut
                if var_j == 1:  # Augmentation
                    valid_j = data[k, attr_j] < data[l, attr_j]
                else:  # Diminution
                    valid_j = data[k, attr_j] > data[l, attr_j]
                
                if valid_i and valid_j:
                    valid_pairs += 1
        
        total_pairs = n * (n - 1) // 2
        return valid_pairs / total_pairs if total_pairs > 0 else 0
    
    def get_patterns(self):
        """Retourne les motifs trouvés."""
        return sorted(self.patterns, key=lambda x: x['support'], reverse=True)

# Exemple avec nos données manuelles
def example_simple_gradual_patterns():
    """Exemple simple de motifs graduels."""
    
    print("=" * 60)
    print("EXEMPLE SIMPLE DE MOTIFS GRADUELS")
    print("=" * 60)
    
    # Nos données d'exemple
    data = np.array([
        [100, 2, 8],   # C1
        [150, 4, 9],   # C2
        [80, 1, 6],    # C3
        [200, 5, 9],   # C4
        [120, 3, 7]    # C5
    ])
    
    # Noms des attributs pour une meilleure lisibilité
    attribute_names = ['Dépenses', 'Fréquence', 'Satisfaction']
    
    # Créer et exécuter le mineur
    miner = GradualPatternMiner(min_support=0.3)
    miner.fit(data)
    
    # Afficher les résultats avec les noms réels
    print("\n" + "=" * 50)
    print("RÉSULTATS DES MOTIFS GRADUELS")
    print("=" * 50)
    
    patterns = miner.get_patterns()
    
    for i, pattern in enumerate(patterns, 1):
        attr_i, attr_j = pattern['attributs']
        var_i, var_j = pattern['variations']
        
        # Convertir en noms lisibles
        name_i = f"{attribute_names[attr_i]}{'+' if var_i == 1 else '-'}"
        name_j = f"{attribute_names[attr_j]}{'+' if var_j == 1 else '-'}"
        
        print(f"\nMotif {i}: {name_i} et {name_j}")
        print(f"  Support: {pattern['support']:.1%}")
        print(f"  Interprétation: Quand {attribute_names[attr_i]} ", end="")
        print(f"{'augmente' if var_i == 1 else 'diminue'}, {attribute_names[attr_j]} ", end="")
        print(f"{'augmente' if var_j == 1 else 'diminue'} aussi")
    
    return miner, patterns

# Version 2 : Implémentation plus avancée avec recherche exhaustive
class AdvancedGradualPatternMiner:
    """Mineur de motifs graduels avec recherche exhaustive."""
    
    def __init__(self, min_support=0.3, max_pattern_size=3):
        self.min_support = min_support
        self.max_pattern_size = max_pattern_size
        self.patterns = []
        self.attribute_names = []
        
    def fit(self, data, attribute_names=None):
        """
        Trouve les motifs graduels de taille 1 à max_pattern_size.
        """
        if isinstance(data, pd.DataFrame):
            self.attribute_names = data.columns.tolist()
            data = data.values
        else:
            data = np.array(data)
            if attribute_names:
                self.attribute_names = attribute_names
            else:
                self.attribute_names = [f"Attribut_{i}" for i in range(data.shape[1])]
        
        n_objects, n_attributes = data.shape
        
        print(f"Analyse des motifs graduels")
        print(f"  - {n_objects} objets, {n_attributes} attributs")
        print(f"  - Support minimum: {self.min_support:.0%}")
        print(f"  - Taille maximum des motifs: {self.max_pattern_size}")
        print("-" * 60)
        
        # Générer tous les motifs possibles
        all_patterns = self._generate_all_patterns(n_attributes)
        
        # Évaluer chaque motif
        for pattern in all_patterns:
            support = self._evaluate_pattern(data, pattern)
            
            if support >= self.min_support:
                self.patterns.append({
                    'pattern': pattern,
                    'support': support
                })
        
        # Trier par support décroissant
        self.patterns.sort(key=lambda x: x['support'], reverse=True)
        
        return self
    
    def _generate_all_patterns(self, n_attributes):
        """Génère tous les motifs possibles."""
        patterns = []
        
        # Pour chaque taille de 1 à max_pattern_size
        for size in range(1, min(self.max_pattern_size, n_attributes) + 1):
            # Pour chaque combinaison d'attributs
            for attr_indices in combinations(range(n_attributes), size):
                # Pour chaque combinaison de variations (+/-)
                # 2^size possibilités
                for var_mask in range(2**size):
                    pattern = []
                    for i in range(size):
                        attr = attr_indices[i]
                        variation = '+' if (var_mask >> i) & 1 else '-'
                        pattern.append((attr, variation))
                    patterns.append(pattern)
        
        return patterns
    
    def _evaluate_pattern(self, data, pattern):
        """Évalue le support d'un motif donné."""
        n = len(data)
        valid_pairs = 0
        
        for i in range(n):
            for j in range(i+1, n):
                valid = True
                
                for attr_idx, variation in pattern:
                    if variation == '+':
                        # Vérifier augmentation
                        if data[i, attr_idx] >= data[j, attr_idx]:
                            valid = False
                            break
                    else:  # variation == '-'
                        # Vérifier diminution
                        if data[i, attr_idx] <= data[j, attr_idx]:
                            valid = False
                            break
                
                if valid:
                    valid_pairs += 1
        
        total_pairs = n * (n - 1) // 2
        return valid_pairs / total_pairs if total_pairs > 0 else 0
    
    def display_results(self, top_k=10):
        """Affiche les résultats de manière lisible."""
        print("\n" + "=" * 60)
        print(f"TOP {min(top_k, len(self.patterns))} MOTIFS GRADUELS")
        print("=" * 60)
        
        for i, pattern_info in enumerate(self.patterns[:top_k], 1):
            pattern = pattern_info['pattern']
            support = pattern_info['support']
            
            # Formatter le motif
            pattern_str = " et ".join(
                [f"{self.attribute_names[attr]}{var}" 
                 for attr, var in pattern]
            )
            
            print(f"\n{i}. {pattern_str}")
            print(f"   Support: {support:.1%}")
            
            # Interprétation
            if len(pattern) == 1:
                attr, var = pattern[0]
                print(f"   Interprétation: {self.attribute_names[attr]} ", end="")
                print(f"{'augmente' if var == '+' else 'diminue'} de manière cohérente")
            else:
                print(f"   Interprétation: Quand ", end="")
                for attr, var in pattern[:-1]:
                    print(f"{self.attribute_names[attr]} ", end="")
                    print(f"{'augmente' if var == '+' else 'diminue'}, ", end="")
                last_attr, last_var = pattern[-1]
                print(f"{self.attribute_names[last_attr]} ", end="")
                print(f"{'augmente' if last_var == '+' else 'diminue'} aussi")

# Exemple avec données réelles simulées
def example_realistic_data():
    """Exemple avec des données plus réalistes."""
    
    print("\n" + "=" * 60)
    print("EXEMPLE AVEC DONNÉES RÉALISTES")
    print("=" * 60)
    
    np.random.seed(42)
    
    # Créer des données corrélées
    n = 100  # 100 clients
    
    # Générer des données corrélées
    base = np.random.normal(50, 10, n)  # Variable de base
    
    # Revenu corrélé avec la base
    revenu = base + np.random.normal(0, 5, n)
    
    # Dépenses fortement corrélées avec le revenu
    depenses = 0.7 * revenu + np.random.normal(0, 8, n)
    
    # Épargne inversement corrélée avec les dépenses
    epargne = 100 - 0.5 * depenses + np.random.normal(0, 10, n)
    
    # Âge (peu corrélé)
    age = np.random.normal(40, 10, n)
    
    # Créer le DataFrame
    data = pd.DataFrame({
        'Revenu': np.maximum(revenu, 20),
        'Dépenses': np.maximum(depenses, 10),
        'Épargne': np.maximum(epargne, 0),
        'Âge': np.maximum(age, 18)
    })
    
    print("Données générées (5 premières lignes):")
    print(data.head())
    print(f"\nCorrélations:")
    print(data.corr().round(2))
    
    # Appliquer le mineur de motifs graduels
    miner = AdvancedGradualPatternMiner(min_support=0.4, max_pattern_size=3)
    miner.fit(data)
    
    # Afficher les résultats
    miner.display_results(top_k=10)
    
    return miner, data

# Exemple 3 : Visualisation des motifs
def visualize_patterns(data, patterns, attribute_names):
    """Visualise les motifs graduels."""
    import matplotlib.pyplot as plt
    
    print("\n" + "=" * 60)
    print("VISUALISATION DES MOTIFS")
    print("=" * 60)
    
    # Prendre quelques motifs intéressants
    interesting_patterns = []
    for pattern_info in patterns[:5]:  # Top 5
        if len(pattern_info['pattern']) == 2:  # Seulement les motifs à 2 attributs
            interesting_patterns.append(pattern_info)
    
    if not interesting_patterns:
        print("Aucun motif à 2 attributs à visualiser")
        return
    
    n_patterns = len(interesting_patterns)
    fig, axes = plt.subplots(1, n_patterns, figsize=(5*n_patterns, 5))
    
    if n_patterns == 1:
        axes = [axes]
    
    for idx, (ax, pattern_info) in enumerate(zip(axes, interesting_patterns)):
        pattern = pattern_info['pattern']
        support = pattern_info['support']
        
        attr1_idx, var1 = pattern[0]
        attr2_idx, var2 = pattern[1]
        
        attr1_name = attribute_names[attr1_idx]
        attr2_name = attribute_names[attr2_idx]
        
        # Tracer les points
        ax.scatter(data[:, attr1_idx], data[:, attr2_idx], alpha=0.6)
        ax.set_xlabel(attr1_name)
        ax.set_ylabel(attr2_name)
        
        # Ajouter une ligne de tendance
        if var1 == '+' and var2 == '+':
            # Relation positive
            color = 'green'
            trend = "Positive"
        elif var1 == '+' and var2 == '-':
            # Relation négative
            color = 'red'
            trend = "Négative"
        elif var1 == '-' and var2 == '+':
            color = 'red'
            trend = "Négative"
        else:  # var1 == '-' and var2 == '-'
            color = 'green'
            trend = "Positive"
        
        # Ajuster une ligne de régression
        z = np.polyfit(data[:, attr1_idx], data[:, attr2_idx], 1)
        p = np.poly1d(z)
        x_range = np.linspace(data[:, attr1_idx].min(), data[:, attr1_idx].max(), 100)
        ax.plot(x_range, p(x_range), color=color, linewidth=2, 
                label=f"Tendance {trend}\nSupport: {support:.1%}")
        
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_title(f"{attr1_name}{var1} et {attr2_name}{var2}")
    
    plt.tight_layout()
    plt.show()

# Exemple 4 : Application pratique - Analyse de marché
def market_analysis_example():
    """Exemple d'analyse de marché avec motifs graduels."""
    
    print("\n" + "=" * 60)
    print("ANALYSE DE MARCHÉ AVEC MOTIFS GRADUELS")
    print("=" * 60)
    
    # Simuler des données de marché
    np.random.seed(123)
    n_products = 50
    
    data = pd.DataFrame({
        'Prix': np.random.uniform(10, 100, n_products),
        'Quantité_vendue': np.random.uniform(100, 1000, n_products),
        'Marge': np.random.uniform(0.1, 0.5, n_products),
        'Stock': np.random.uniform(50, 500, n_products),
        'Clients_fidèles': np.random.uniform(0, 1, n_products)  # Proportion
    })
    
    # Ajouter des corrélations réalistes
    # Prix élevé → Marge élevée
    data['Marge'] = 0.1 + 0.004 * data['Prix'] + np.random.normal(0, 0.05, n_products)
    
    # Prix élevé → Quantité vendue faible (loi de la demande)
    data['Quantité_vendue'] = 1000 - 5 * data['Prix'] + np.random.normal(0, 200, n_products)
    
    # Stock élevé → Clients fidèles (disponibilité)
    data['Clients_fidèles'] = 0.3 + 0.001 * data['Stock'] + np.random.normal(0, 0.1, n_products)
    
    print("Données de marché (5 premiers produits):")
    print(data.head())
    
    print("\nAnalyse des corrélations:")
    corr_matrix = data.corr()
    print(corr_matrix.round(2))
    
    # Chercher des motifs graduels
    miner = AdvancedGradualPatternMiner(min_support=0.35, max_pattern_size=2)
    miner.fit(data)
    
    print("\n" + "=" * 50)
    print("INSIGHTS DE MARCHÉ TROUVÉS")
    print("=" * 50)
    
    # Filtrer et organiser les insights
    insights = []
    
    for pattern_info in miner.patterns[:15]:  # Prendre plus pour filtrer
        pattern = pattern_info['pattern']
        support = pattern_info['support']
        
        if len(pattern) == 2 and support > 0.4:
            attr1, var1 = pattern[0]
            attr2, var2 = pattern[1]
            
            attr1_name = miner.attribute_names[attr1]
            attr2_name = miner.attribute_names[attr2]
            
            # Catégoriser les insights
            if "Prix" in attr1_name and "Quantité" in attr2_name:
                if var1 == '+' and var2 == '-':
                    category = "LOI DE LA DEMANDE"
                elif var1 == '+' and var2 == '+':
                    category = "PRODUIT DE LUXE"
                    
            elif "Prix" in attr1_name and "Marge" in attr2_name:
                if var1 == '+' and var2 == '+':
                    category = "STRATÉGIE PRIX"
                    
            elif "Stock" in attr1_name and "Clients" in attr2_name:
                if var1 == '+' and var2 == '+':
                    category = "FIDÉLISATION"
                    
            else:
                category = "AUTRE"
            
            insights.append({
                'category': category,
                'pattern': f"{attr1_name}{var1} et {attr2_name}{var2}",
                'support': support,
                'interpretation': f"Quand {attr1_name} {'augmente' if var1 == '+' else 'diminue'}, "
                                  f"{attr2_name} {'augmente' if var2 == '+' else 'diminue'}"
            })
    
    # Afficher par catégorie
    categories = {}
    for insight in insights:
        cat = insight['category']
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(insight)
    
    for category, cat_insights in categories.items():
        print(f"\n{category}:")
        print("-" * 40)
        for insight in cat_insights:
            print(f"  • {insight['pattern']} (support: {insight['support']:.1%})")
            print(f"    → {insight['interpretation']}")
    
    return miner, data, insights

# Fonction principale
def main():
    """Exécute tous les exemples de motifs graduels."""
    print("=" * 60)
    print("MOTIFS GRADUELS - DÉTECTION DE TENDANCES ÉVOLUTIVES")
    print("=" * 60)
    
    # Exemple 1 : Simple (pédagogique)
    print("\n1. EXEMPLE PÉDAGOGIQUE (notre exemple manuel)")
    miner_simple, patterns_simple = example_simple_gradual_patterns()
    
    # Exemple 2 : Données réalistes
    print("\n2. DONNÉES RÉALISTES CORRÉLÉES")
    miner_advanced, data_realistic = example_realistic_data()
    
    # Exemple 3 : Analyse de marché
    print("\n3. APPLICATION : ANALYSE DE MARCHÉ")
    miner_market, data_market, insights = market_analysis_example()
    
    print("\n" + "=" * 60)
    print("CONCLUSION")
    print("=" * 60)
    print("""
    Les motifs graduels permettent de découvrir des tendances évolutives 
    dans les données numériques. Contrairement aux règles d'association,
    ils capturent des relations de variation ("quand X augmente, Y diminue").
    
    Applications :
    1. Analyse de marché (prix vs demande)
    2. Étude économique (inflation vs chômage)
    3. Médecine (symptômes vs progression maladie)
    4. Finance (actions corrélées)
    
    Avantages :
    - Capture des relations directionnelles
    - Pas besoin de discrétisation
    - Interprétation intuitive
    
    Limitations :
    - Complexité computationnelle
    - Sensible au bruit dans les données
    - Nécessite des données numériques
    """)

if __name__ == "__main__":
    main()
