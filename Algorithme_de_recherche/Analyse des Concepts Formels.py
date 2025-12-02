import numpy as np
from itertools import combinations, chain

class FormalContext:
    """Contexte formel pour l'analyse des concepts."""
    
    def __init__(self, objects, attributes, incidence):
        """
        Initialise un contexte formel.
        
        Parameters:
        -----------
        objects : list, noms des objets
        attributes : list, noms des attributs
        incidence : array 2D booléen, incidence[i,j] = objet i a attribut j
        """
        self.objects = objects
        self.attributes = attributes
        self.incidence = np.array(incidence, dtype=bool)
        
        # Mappings pour accès rapide
        self.obj_to_idx = {obj: i for i, obj in enumerate(objects)}
        self.attr_to_idx = {attr: i for i, attr in enumerate(attributes)}
        
    def object_derivation(self, object_set):
        """A' : attributs communs à un ensemble d'objets."""
        if not object_set:
            return set(self.attributes)  # Tous les attributs
        
        # Convertir les noms en indices
        indices = [self.obj_to_idx[obj] for obj in object_set]
        
        # Intersection des attributs
        common_attributes = np.all(self.incidence[indices, :], axis=0)
        
        # Convertir en noms
        result = {self.attributes[i] for i in np.where(common_attributes)[0]}
        return result
    
    def attribute_derivation(self, attribute_set):
        """B' : objets ayant tous les attributs."""
        if not attribute_set:
            return set(self.objects)  # Tous les objets
        
        # Convertir les noms en indices
        indices = [self.attr_to_idx[attr] for attr in attribute_set]
        
        # Intersection des objets
        common_objects = np.all(self.incidence[:, indices], axis=1)
        
        # Convertir en noms
        result = {self.objects[i] for i in np.where(common_objects)[0]}
        return result
    
    def is_closed(self, object_set, attribute_set):
        """Vérifie si (A, B) est un concept fermé."""
        A_prime = self.object_derivation(object_set)
        B_prime = self.attribute_derivation(attribute_set)
        
        return (set(attribute_set) == A_prime and 
                set(object_set) == B_prime)
    
    def closure_objects(self, object_set):
        """Fermeture d'un ensemble d'objets : A → A''."""
        attributes = self.object_derivation(object_set)
        closed_objects = self.attribute_derivation(attributes)
        return closed_objects
    
    def closure_attributes(self, attribute_set):
        """Fermeture d'un ensemble d'attributs : B → B''."""
        objects = self.attribute_derivation(attribute_set)
        closed_attributes = self.object_derivation(objects)
        return closed_attributes

class FormalConceptAnalyzer:
    """Analyseur de concepts formels."""
    
    def __init__(self, context):
        self.context = context
        self.concepts = []
        self.lattice = {}
        
    def compute_all_concepts(self, method='next_closure'):
        """Calcule tous les concepts formels."""
        print("=" * 60)
        print("ANALYSE DES CONCEPTS FORMELS")
        print("=" * 60)
        
        n_objects = len(self.context.objects)
        n_attributes = len(self.context.attributes)
        
        print(f"Objets: {self.context.objects}")
        print(f"Attributs: {self.context.attributes}")
        print(f"Taille du contexte: {n_objects} × {n_attributes}")
        print("-" * 40)
        
        if method == 'naive':
            self._compute_concepts_naive()
        elif method == 'next_closure':
            self._compute_concepts_next_closure()
        
        print(f"\nConcepts trouvés: {len(self.concepts)}")
        
        # Trier par taille d'extension (décroissant)
        self.concepts.sort(key=lambda c: len(c[0]), reverse=True)
        
        return self.concepts
    
    def _compute_concepts_naive(self):
        """Méthode naïve (force brute) - pour petits contextes."""
        n_objects = len(self.context.objects)
        
        # Générer tous les sous-ensembles d'objets possibles
        all_object_subsets = chain.from_iterable(
            combinations(self.context.objects, r) 
            for r in range(n_objects + 1)
        )
        
        self.concepts = []
        
        for obj_set in all_object_subsets:
            obj_set = set(obj_set)
            
            # Calculer l'intension
            intent = self.context.object_derivation(obj_set)
            
            # Calculer la fermeture
            closed_extent = self.context.attribute_derivation(intent)
            
            # Vérifier la fermeture
            if obj_set == closed_extent:
                concept = (frozenset(closed_extent), frozenset(intent))
                if concept not in self.concepts:
                    self.concepts.append(concept)
    
    def _compute_concepts_next_closure(self):
        """Algorithme Next-Closure (plus efficace)."""
        # Concept infimum (bottom)
        bottom_extent = set()
        bottom_intent = self.context.object_derivation(bottom_extent)
        bottom_concept = (frozenset(bottom_extent), frozenset(bottom_intent))
        self.concepts = [bottom_concept]
        
        # Générer récursivement
        current = bottom_concept
        
        while True:
            # Générer le prochain concept
            next_concept = self._next_closure(current[1])
            
            if next_concept is None:
                break
            
            self.concepts.append(next_concept)
            current = next_concept
        
        # Ajouter le concept suprême (top)
        top_intent = set()
        top_extent = self.context.attribute_derivation(top_intent)
        top_concept = (frozenset(top_extent), frozenset(top_intent))
        
        if top_concept not in self.concepts:
            self.concepts.append(top_concept)
    
    def _next_closure(self, intent):
        """Implémente l'algorithme Next-Closure."""
        attributes = self.context.attributes
        m = len(attributes)
        
        # Pour chaque attribut dans l'ordre inverse
        for i in range(m - 1, -1, -1):
            attr = attributes[i]
            
            if attr in intent:
                continue
            
            # Créer un nouvel ensemble d'attributs candidat
            new_intent = set(intent)
            new_intent.add(attr)
            
            # Fermeture
            closed_extent = self.context.attribute_derivation(new_intent)
            closed_intent = self.context.object_derivation(closed_extent)
            
            # Vérifier si c'est le "next" dans l'ordre lexicographique
            is_next = True
            for j in range(i):
                attr_j = attributes[j]
                if attr_j not in closed_intent and attr_j in intent:
                    is_next = False
                    break
            
            if is_next:
                return (frozenset(closed_extent), frozenset(closed_intent))
        
        return None
    
    def display_concepts(self, max_display=10):
        """Affiche les concepts trouvés."""
        print(f"\n{'='*60}")
        print(f"CONCEPTS FORMELS ({len(self.concepts)} trouvés)")
        print('='*60)
        
        for i, (extent, intent) in enumerate(self.concepts[:max_display], 1):
            extent_str = ", ".join(sorted(extent)) if extent else "∅"
            intent_str = ", ".join(sorted(intent)) if intent else "∅"
            
            print(f"\nConcept {i}:")
            print(f"  Extension (objets): {{{extent_str}}}")
            print(f"  Intension (attributs): {{{intent_str}}}")
            print(f"  Taille: {len(extent)} objets, {len(intent)} attributs")
        
        if len(self.concepts) > max_display:
            print(f"\n... et {len(self.concepts) - max_display} autres concepts")
    
    def build_concept_lattice(self):
        """Construit le treillis des concepts."""
        print(f"\n{'='*60}")
        print("CONSTRUCTION DU TREILLIS")
        print('='*60)
        
        # Créer un mapping pour accès rapide
        concept_dict = {}
        for i, (extent, intent) in enumerate(self.concepts):
            concept_dict[(frozenset(extent), frozenset(intent))] = i
        
        # Initialiser le treillis
        self.lattice = {
            'nodes': [],
            'edges': []
        }
        
        # Ajouter les nœuds
        for extent, intent in self.concepts:
            self.lattice['nodes'].append({
                'extent': extent,
                'intent': intent,
                'label': self._concept_label(extent, intent)
            })
        
        # Ajouter les arêtes (relations de sous-concepts)
        n = len(self.concepts)
        for i in range(n):
            for j in range(n):
                if i != j:
                    # i est sous-concept de j si extent_i ⊆ extent_j
                    if self.concepts[i][0].issubset(self.concepts[j][0]):
                        # Vérifier qu'il n'y a pas de concept intermédiaire
                        is_direct = True
                        for k in range(n):
                            if (k != i and k != j and
                                self.concepts[i][0].issubset(self.concepts[k][0]) and
                                self.concepts[k][0].issubset(self.concepts[j][0])):
                                is_direct = False
                                break
                        
                        if is_direct:
                            self.lattice['edges'].append((i, j))
        
        print(f"Treillis construit: {len(self.lattice['nodes'])} nœuds, "
              f"{len(self.lattice['edges'])} arêtes")
        
        return self.lattice
    
    def _concept_label(self, extent, intent):
        """Crée un label lisible pour un concept."""
        if not extent and not intent:
            return "(∅, {tous})"
        elif not extent:
            return f"(∅, {{{', '.join(sorted(intent))}}})"
        elif not intent:
            return f"({{{', '.join(sorted(extent))}}}, ∅)"
        else:
            return f"({{{', '.join(sorted(extent))}}}, {{{', '.join(sorted(intent))}}})"
    
    def visualize_lattice(self):
        """Visualise le treillis (simplifié)."""
        try:
            import networkx as nx
            import matplotlib.pyplot as plt
            
            G = nx.DiGraph()
            
            # Ajouter les nœuds avec labels
            for i, node in enumerate(self.lattice['nodes']):
                G.add_node(i, label=node['label'])
            
            # Ajouter les arêtes (inversées pour avoir top→bottom)
            for source, target in self.lattice['edges']:
                G.add_edge(target, source)  # Inverser pour hiérarchie
            
            # Dessiner
            plt.figure(figsize=(12, 8))
            pos = nx.spring_layout(G, seed=42)
            
            nx.draw(G, pos, with_labels=False, node_size=500, 
                   node_color='lightblue', arrows=True)
            
            # Ajouter les labels
            labels = nx.get_node_attributes(G, 'label')
            pos_labels = {}
            for node, coords in pos.items():
                pos_labels[node] = (coords[0], coords[1] - 0.05)
            
            nx.draw_networkx_labels(G, pos_labels, labels, font_size=8)
            
            plt.title("Treillis des Concepts Formels", fontsize=14)
            plt.axis('off')
            plt.tight_layout()
            plt.show()
            
        except ImportError:
            print("Installation requise: pip install networkx matplotlib")
            self._print_lattice_text()

# Exemple avec nos données animaux
def example_animal_fca():
    """Exemple d'analyse des concepts avec les animaux."""
    
    print("=" * 60)
    print("ANALYSE DES CONCEPTS: ANIMAUX")
    print("=" * 60)
    
    # Définir le contexte
    objects = ['Chat', 'Chien', 'Oiseau', 'Poisson', 'Serpent']
    attributes = ['Mammifère', 'Volant', 'Nageant', 'Carnivore', 'Domestique']
    
    # Table d'incidence (0 = non, 1 = oui)
    incidence = [
        [1, 0, 0, 1, 1],  # Chat
        [1, 0, 1, 1, 1],  # Chien
        [0, 1, 0, 0, 0],  # Oiseau
        [0, 0, 1, 0, 0],  # Poisson
        [0, 0, 0, 1, 0]   # Serpent
    ]
    
    # Créer le contexte
    context = FormalContext(objects, attributes, incidence)
    
    # Afficher la table
    print("\nTable d'incidence:")
    print("     " + " ".join(f"{a:10}" for a in attributes))
    for i, obj in enumerate(objects):
        row = " ".join(f"{'X' if val else '.':10}" for val in incidence[i])
        print(f"{obj:7} {row}")
    
    # Tester les opérations de dérivation
    print("\n" + "-"*40)
    print("TESTS DE DÉRIVATION:")
    print("-"*40)
    
    # Exemple 1: Attributs du Chat
    print(f"\nAttributs de {{Chat}}: {context.object_derivation({'Chat'})}")
    
    # Exemple 2: Objets carnivores
    print(f"Objets carnivores ({{Carnivore}}): {context.attribute_derivation({'Carnivore'})}")
    
    # Exemple 3: Attributs communs Chat et Chien
    print(f"Attributs communs Chat et Chien: {context.object_derivation({'Chat', 'Chien'})}")
    
    # Analyser les concepts
    analyzer = FormalConceptAnalyzer(context)
    concepts = analyzer.compute_all_concepts(method='naive')
    
    # Afficher les concepts
    analyzer.display_concepts(max_display=10)
    
    # Construire le treillis
    lattice = analyzer.build_concept_lattice()
    
    # Afficher le treillis en texte
    print(f"\n{'='*60}")
    print("TREILLIS DES CONCEPTS (représentation textuelle)")
    print('='*60)
    
    # Trier par taille d'extension (top first)
    sorted_concepts = sorted(concepts, key=lambda c: len(c[0]), reverse=True)
    
    for i, (extent, intent) in enumerate(sorted_concepts):
        level = len(extent)  # Niveau approximatif dans le treillis
        indent = "  " * (5 - level)
        
        extent_str = ", ".join(sorted(extent)) if extent else "∅"
        intent_str = ", ".join(sorted(intent)) if intent else "∅"
        
        print(f"{indent}Niveau {level}: ({extent_str}) | ({intent_str})")
    
    return analyzer, concepts

# Version 2 : FCA avec règles d'implication
class FCARuleMiner:
    """Extraction de règles d'implication à partir des concepts."""
    
    def __init__(self, context, concepts):
        self.context = context
        self.concepts = concepts
        
    def extract_implications(self, min_support=0.3, min_confidence=0.7):
        """Extrait les règles d'implication."""
        print(f"\n{'='*60}")
        print("EXTRACTION DE RÈGLES D'IMPLICATION")
        print('='*60)
        
        implications = []
        n_objects = len(self.context.objects)
        
        # Pour chaque concept
        for extent, intent in self.concepts:
            if len(intent) >= 2:  # Au moins 2 attributs pour avoir une implication
                # Générer toutes les implications possibles
                intent_list = list(intent)
                
                for i in range(len(intent_list)):
                    for j in range(i+1, len(intent_list)):
                        attr1 = intent_list[i]
                        attr2 = intent_list[j]
                        
                        # Support = taille de l'extension
                        support = len(extent) / n_objects
                        
                        if support >= min_support:
                            # Règle: attr1 → attr2
                            conf1 = self._confidence({attr1}, {attr2})
                            
                            if conf1 >= min_confidence:
                                implications.append({
                                    'antecedent': {attr1},
                                    'consequent': {attr2},
                                    'support': support,
                                    'confidence': conf1
                                })
                            
                            # Règle inverse: attr2 → attr1
                            conf2 = self._confidence({attr2}, {attr1})
                            
                            if conf2 >= min_confidence:
                                implications.append({
                                    'antecedent': {attr2},
                                    'consequent': {attr1},
                                    'support': support,
                                    'confidence': conf2
                                })
        
        # Trier par confiance décroissante
        implications.sort(key=lambda x: x['confidence'], reverse=True)
        
        print(f"\nRègles trouvées: {len(implications)}")
        print("-" * 40)
        
        for i, rule in enumerate(implications[:10], 1):
            ant = ", ".join(rule['antecedent'])
            cons = ", ".join(rule['consequent'])
            print(f"Règle {i}: {ant} → {cons}")
            print(f"  Support: {rule['support']:.1%}, "
                  f"Confiance: {rule['confidence']:.1%}")
        
        return implications
    
    def _confidence(self, antecedent, consequent):
        """Calcule la confiance d'une règle A → B."""
        # Objets ayant A
        objects_A = self.context.attribute_derivation(antecedent)
        
        if not objects_A:
            return 0.0
        
        # Objets ayant A et B
        objects_AB = self.context.attribute_derivation(antecedent.union(consequent))
        
        return len(objects_AB) / len(objects_A)

# Exemple 2 : FCA sur données clients
def customer_fca_example():
    """Exemple d'analyse de clients avec FCA."""
    
    print("\n" + "=" * 60)
    print("ANALYSE DE CLIENTS (FCA APPLIQUÉE)")
    print("=" * 60)
    
    # Contexte: Clients et leurs achats
    customers = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8']
    products = ['Lait', 'Pain', 'Beurre', 'Œufs', 'Fromage', 'Jus']
    
    # Données d'achat
    incidence = [
        [1, 1, 1, 0, 0, 0],  # C1: Lait, Pain, Beurre
        [1, 1, 0, 1, 0, 0],  # C2: Lait, Pain, Œufs
        [0, 1, 1, 0, 1, 0],  # C3: Pain, Beurre, Fromage
        [1, 0, 0, 1, 0, 1],  # C4: Lait, Œufs, Jus
        [0, 1, 0, 1, 0, 0],  # C5: Pain, Œufs
        [1, 1, 0, 0, 0, 1],  # C6: Lait, Pain, Jus
        [0, 0, 1, 1, 1, 0],  # C7: Beurre, Œufs, Fromage
        [1, 0, 1, 0, 1, 0]   # C8: Lait, Beurre, Fromage
    ]
    
    # Créer le contexte
    context = FormalContext(customers, products, incidence)
    
    print("\nContexte clients-produits:")
    print("       " + " ".join(f"{p:8}" for p in products))
    for i, cust in enumerate(customers):
        row = " ".join(f"{'X' if val else '.':8}" for val in incidence[i])
        print(f"{cust:6} {row}")
    
    # Analyser
    analyzer = FormalConceptAnalyzer(context)
    concepts = analyzer.compute_all_concepts()
    
    print(f"\nNombre total de concepts: {len(concepts)}")
    
    # Afficher les concepts intéressants
    print(f"\n{'='*40}")
    print("CONCEPTS INTÉRESSANTS (segments clients)")
    print('='*40)
    
    interesting_concepts = []
    for extent, intent in concepts:
        if len(extent) >= 2 and len(intent) >= 2:  # Concepts non triviaux
            interesting_concepts.append((extent, intent))
    
    # Trier par taille d'extension
    interesting_concepts.sort(key=lambda c: len(c[0]), reverse=True)
    
    for i, (extent, intent) in enumerate(interesting_concepts[:5], 1):
        customers_str = ", ".join(sorted(extent))
        products_str = ", ".join(sorted(intent))
        
        print(f"\nSegment {i}:")
        print(f"  Clients: {customers_str}")
        print(f"  Produits achetés: {products_str}")
        print(f"  Taille: {len(extent)} clients")
        
        # Interprétation
        if 'Lait' in intent and 'Pain' in intent:
            print(f"  💡 Insight: Ces clients achètent systématiquement Lait et Pain ensemble")
        if 'Beurre' in intent and 'Fromage' in intent:
            print(f"  💡 Insight: Clients produits laitiers (hors lait)")
    
    # Extraire des règles d'implication
    rule_miner = FCARuleMiner(context, concepts)
    rules = rule_miner.extract_implications(min_support=0.25, min_confidence=0.8)
    
    # Recommandations basées sur les règles
    print(f"\n{'='*40}")
    print("RECOMMANDATIONS COMMERCIALES")
    print('='*40)
    
    # Pour chaque client, recommander des produits
    for customer in customers[:3]:  # 3 premiers clients
        print(f"\nRecommandations pour {customer}:")
        
        # Produits déjà achetés
        bought = set()
        cust_idx = context.obj_to_idx[customer]
        for j, attr in enumerate(products):
            if incidence[cust_idx][j]:
                bought.add(attr)
        
        if bought:
            print(f"  Déjà achetés: {', '.join(sorted(bought))}")
            
            # Chercher des règles applicables
            recommendations = set()
            for rule in rules[:5]:  # Top 5 règles
                if rule['antecedent'].issubset(bought):
                    new_products = rule['consequent'] - bought
                    recommendations.update(new_products)
            
            if recommendations:
                print(f"  Recommandations: {', '.join(sorted(recommendations))}")
            else:
                print(f"  Aucune recommandation forte trouvée")
        else:
            print(f"  Aucun achat enregistré")
    
    return analyzer, concepts, rules

# Version 3 : FCA avec visualisation avancée
class AdvancedFCAVisualizer:
    """Visualisation avancée pour FCA."""
    
    @staticmethod
    def draw_concept_matrix(concepts, context):
        """Dessine la matrice des concepts."""
        import matplotlib.pyplot as plt
        
        # Préparer les données
        n_concepts = len(concepts)
        n_objects = len(context.objects)
        n_attributes = len(context.attributes)
        
        # Créer la matrice conceptuelle
        concept_matrix = np.zeros((n_concepts, n_objects + n_attributes), dtype=int)
        
        for i, (extent, intent) in enumerate(concepts):
            # Partie objets
            for obj in extent:
                j = context.objects.index(obj)
                concept_matrix[i, j] = 1
            
            # Partie attributs
            for attr in intent:
                j = n_objects + context.attributes.index(attr)
                concept_matrix[i, j] = 1
        
        # Dessiner
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Heatmap
        im = ax.imshow(concept_matrix, cmap='Blues', aspect='auto')
        
        # Grid
        ax.set_xticks(np.arange(n_objects + n_attributes))
        ax.set_yticks(np.arange(n_concepts))
        
        # Labels
        x_labels = context.objects + context.attributes
        ax.set_xticklabels(x_labels, rotation=45, ha='right')
        ax.set_yticklabels([f"C{i+1}" for i in range(n_concepts)])
        
        # Ligne de séparation objets/attributs
        ax.axvline(x=n_objects - 0.5, color='red', linewidth=2)
        
        # Titres
        ax.set_title("Matrice des Concepts Formels", fontsize=14)
        ax.set_xlabel("Objets | Attributs", fontsize=12)
        ax.set_ylabel("Concepts", fontsize=12)
        
        plt.colorbar(im, ax=ax, label='Présence')
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def draw_line_diagram(concepts, context):
        """Diagramme en ligne de Hasse simplifié."""
        try:
            import networkx as nx
            import matplotlib.pyplot as plt
            
            G = nx.DiGraph()
            
            # Créer les nœuds
            for i, (extent, intent) in enumerate(concepts):
                # Label court
                if len(extent) <= 2 and len(intent) <= 2:
                    ext_str = "".join(sorted([o[0] for o in extent])) if extent else "∅"
                    int_str = "".join(sorted([a[0] for a in intent])) if intent else "∅"
                    label = f"({ext_str}|{int_str})"
                else:
                    label = f"C{i+1}"
                
                G.add_node(i, label=label, extent=extent, intent=intent)
            
            # Ajouter les arêtes (sous-concepts directs)
            n = len(concepts)
            edges = []
            
            for i in range(n):
                for j in range(n):
                    if i != j:
                        # i sous-concept de j si extent_i ⊆ extent_j
                        if concepts[i][0].issubset(concepts[j][0]):
                            # Vérifier la directité
                            is_direct = True
                            for k in range(n):
                                if (k != i and k != j and
                                    concepts[i][0].issubset(concepts[k][0]) and
                                    concepts[k][0].issubset(concepts[j][0])):
                                    is_direct = False
                                    break
                            
                            if is_direct:
                                edges.append((j, i))  # De plus général à plus spécifique
            
            G.add_edges_from(edges)
            
            # Dessiner
            plt.figure(figsize=(10, 8))
            
            # Position hiérarchique
            pos = {}
            levels = {}
            
            # Assigner des niveaux basés sur la taille de l'extension
            for node in G.nodes():
                extent_size = len(concepts[node][0])
                levels[node] = extent_size
            
            max_level = max(levels.values())
            
            for node in G.nodes():
                level = levels[node]
                # Positionner horizontalement au sein du niveau
                nodes_in_level = [n for n in G.nodes() if levels[n] == level]
                idx = nodes_in_level.index(node)
                total_in_level = len(nodes_in_level)
                
                x = idx - (total_in_level - 1) / 2
                y = max_level - level  # Inverser pour avoir top en haut
                pos[node] = (x, y)
            
            nx.draw(G, pos, with_labels=True, 
                   labels=nx.get_node_attributes(G, 'label'),
                   node_size=800, node_color='lightgreen',
                   font_size=10, font_weight='bold',
                   arrows=False)
            
            plt.title("Diagramme en Ligne de Hasse", fontsize=14)
            plt.axis('off')
            plt.tight_layout()
            plt.show()
            
        except ImportError:
            print("Installation requise: pip install networkx matplotlib")

# Fonction principale
def main():
    """Exécute tous les exemples de FCA."""
    print("=" * 60)
    print("ANALYSE DES CONCEPTS FORMELS (FCA)")
    print("=" * 60)
    
    # Exemple 1 : Animaux
    print("\n1. EXEMPLE PÉDAGOGIQUE: ANIMAUX")
    analyzer_animals, concepts_animals = example_animal_fca()
    
    # Exemple 2 : Clients
    print("\n2. EXEMPLE APPLIQUÉ: SEGMENTATION CLIENTS")
    analyzer_customers, concepts_customers, rules = customer_fca_example()
    
    # Visualisation avancée (si bibliothèques disponibles)
    try:
        print("\n3. VISUALISATION AVANCÉE")
        print("-" * 40)
        
        # Redéfinir le contexte clients pour la visualisation
        customers = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8']
        products = ['Lait', 'Pain', 'Beurre', 'Œufs', 'Fromage', 'Jus']
        incidence = [
            [1, 1, 1, 0, 0, 0],
            [1, 1, 0, 1, 0, 0],
            [0, 1, 1, 0, 1, 0],
            [1, 0, 0, 1, 0, 1],
            [0, 1, 0, 1, 0, 0],
            [1, 1, 0, 0, 0, 1],
            [0, 0, 1, 1, 1, 0],
            [1, 0, 1, 0, 1, 0]
        ]
        
        context = FormalContext(customers, products, incidence)
        analyzer = FormalConceptAnalyzer(context)
        concepts = analyzer.compute_all_concepts()
        
        # Visualiser
        visualizer = AdvancedFCAVisualizer()
        visualizer.draw_line_diagram(concepts, context)
        
    except Exception as e:
        print(f"Visualisation non disponible: {e}")
    
    print("\n" + "=" * 60)
    print("RÉSUMÉ DE L'ANALYSE DES CONCEPTS FORMELS")
    print("=" * 60)
    print("""
    L'Analyse des Concepts Formels (FCA) transforme des données en concepts
    hiérarchisés. C'est une alternative aux méthodes statistiques classiques.
    
    Étapes principales:
    1. Définir le contexte (Objets × Attributs)
    2. Calculer les dérivations (A', B')
    3. Trouver les concepts fermés (A = A'', B = B'')
    4. Construire le treillis de concepts
    5. Extraire des règles d'implication
    
    Applications:
    • Data mining: Découverte de motifs fréquents
    • Connaissances: Organisation de l'information
    • Ontologies: Construction de taxonomies
    • Marketing: Segmentation de clients
    • Pédagogie: Analyse de compétences
    
    Algorithmes clés:
    • Next-Closure: Pour calculer tous les concepts
    • Bordat: Construction incrémentale du treillis
    • Ganter: Algorithme efficace pour grands contextes
    
    Avantages:
    • Résultats interprétables (concepts clairs)
    • Hiérarchie naturelle des connaissances
    • Pas besoin de paramètres (vs clustering)
    
    Limites:
    • Complexité exponentielle dans le pire cas
    • Données binaires seulement (nécessite discrétisation)
    • Peu adapté aux très grands datasets
    """)

if __name__ == "__main__":
    main()
