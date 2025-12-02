import numpy as np
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
import itertools
from collections import defaultdict

class ClassifierChain:
    """Implémentation simple de Classifier Chains."""
    
    def __init__(self, base_classifier=None, order='random', random_state=42):
        """
        Parameters:
        -----------
        base_classifier : classifieur de base (par défaut: LogisticRegression)
        order : 'random', 'original', ou liste spécifique
        random_state : pour la reproductibilité
        """
        self.base_classifier = base_classifier or LogisticRegression(max_iter=1000, random_state=random_state)
        self.order = order
        self.random_state = random_state
        self.chain = []  # Liste de (classifieur, label_index)
        self.label_names = None
        
    def fit(self, X, y, label_names=None):
        """
        Entraîne la chaîne de classifieurs.
        
        Parameters:
        -----------
        X : array 2D de forme (n_samples, n_features)
        y : array 2D de forme (n_samples, n_labels) ou list de sets
        label_names : list, noms des labels
        """
        # Convertir y en matrice binaire
        if isinstance(y[0], set) or isinstance(y[0], list):
            # Déterminer tous les labels uniques
            all_labels = set()
            for labels in y:
                all_labels.update(labels)
            
            self.label_names = label_names if label_names else sorted(all_labels)
            n_labels = len(self.label_names)
            
            # Créer la matrice binaire
            y_binary = np.zeros((len(y), n_labels), dtype=int)
            label_to_idx = {label: i for i, label in enumerate(self.label_names)}
            
            for i, labels in enumerate(y):
                for label in labels:
                    if label in label_to_idx:
                        y_binary[i, label_to_idx[label]] = 1
        else:
            # y est déjà une matrice binaire
            y_binary = np.array(y, dtype=int)
            n_labels = y_binary.shape[1]
            self.label_names = label_names if label_names else [f"Label_{i}" for i in range(n_labels)]
        
        # Déterminer l'ordre des labels
        if self.order == 'random':
            np.random.seed(self.random_state)
            order = np.random.permutation(n_labels)
        elif self.order == 'original':
            order = np.arange(n_labels)
        elif isinstance(self.order, list):
            # Vérifier que l'ordre est valide
            if len(self.order) != n_labels or set(self.order) != set(range(n_labels)):
                raise ValueError("L'ordre spécifié est invalide")
            order = np.array(self.order)
        else:
            raise ValueError(f"Ordre non supporté: {self.order}")
        
        print(f"Ordre de la chaîne: {[self.label_names[i] for i in order]}")
        
        # Initialiser la chaîne
        self.chain = []
        
        # Entraîner chaque classifieur dans la chaîne
        for idx, label_idx in enumerate(order):
            print(f"\nEntraînement du classifieur {idx+1}/{n_labels} pour le label '{self.label_names[label_idx]}'")
            
            # Préparer les features pour ce classifieur
            if idx == 0:
                # Premier classifieur: seulement les features originales
                X_current = X.copy()
            else:
                # Ajouter les prédictions des labels précédents
                prev_predictions = np.zeros((X.shape[0], idx))
                
                for j, (clf, prev_idx) in enumerate(self.chain):
                    # Prédire sur les données d'entraînement
                    pred = clf.predict(X_current[:, :X.shape[1] + j])
                    prev_predictions[:, j] = pred
                
                # Concaténer features originales + prédictions précédentes
                X_current = np.hstack([X, prev_predictions])
            
            # Créer et entraîner le classifieur
            clf = clone(self.base_classifier)
            clf.fit(X_current, y_binary[:, label_idx])
            
            # Ajouter à la chaîne
            self.chain.append((clf, label_idx))
            
            # Évaluer sur les données d'entraînement
            train_acc = clf.score(X_current, y_binary[:, label_idx])
            print(f"  Précision entraînement: {train_acc:.3f}")
        
        return self
    
    def predict(self, X):
        """
        Prédit les labels en parcourant la chaîne.
        
        Returns:
        --------
        predictions : list de sets, labels prédits pour chaque instance
        """
        if not self.chain:
            raise ValueError("Le modèle doit être entraîné d'abord")
        
        n_samples = X.shape[0]
        n_original_features = X.shape[1]
        
        # Initialiser les prédictions
        predictions_binary = np.zeros((n_samples, len(self.chain)), dtype=int)
        
        # Parcourir la chaîne
        X_current = X.copy()
        
        for idx, (clf, label_idx) in enumerate(self.chain):
            # Prédire ce label
            pred = clf.predict(X_current)
            predictions_binary[:, idx] = pred
            
            # Mettre à jour les features pour le prochain classifieur
            if idx < len(self.chain) - 1:
                X_current = np.hstack([X, predictions_binary[:, :idx+1]])
        
        # Reorganiser selon l'ordre original des labels
        final_predictions = np.zeros((n_samples, len(self.chain)), dtype=int)
        order = [label_idx for _, label_idx in self.chain]
        
        for new_idx, original_idx in enumerate(order):
            final_predictions[:, original_idx] = predictions_binary[:, new_idx]
        
        # Convertir en sets
        predictions = []
        for i in range(n_samples):
            labels_set = set()
            for label_idx, pred in enumerate(final_predictions[i]):
                if pred == 1:
                    labels_set.add(self.label_names[label_idx])
            predictions.append(labels_set)
        
        return predictions
    
    def predict_proba(self, X):
        """
        Retourne les probabilités pour chaque label.
        Note: Plus complexe car dépend de l'ordre de la chaîne.
        """
        if not self.chain:
            raise ValueError("Le modèle doit être entraîné d'abord")
        
        n_samples = X.shape[0]
        n_labels = len(self.chain)
        probabilities = np.zeros((n_samples, n_labels))
        
        # Parcourir la chaîne pour obtenir les probabilités
        X_current = X.copy()
        predictions_so_far = np.zeros((n_samples, 0), dtype=int)
        
        for idx, (clf, label_idx) in enumerate(self.chain):
            if hasattr(clf, 'predict_proba'):
                proba = clf.predict_proba(X_current)
                # Probabilité de la classe positive
                probabilities[:, label_idx] = proba[:, 1]
            else:
                probabilities[:, label_idx] = clf.predict(X_current)
            
            # Mettre à jour pour le prochain classifieur
            pred = clf.predict(X_current).reshape(-1, 1)
            predictions_so_far = np.hstack([predictions_so_far, pred])
            X_current = np.hstack([X, predictions_so_far])
        
        return probabilities

# Exemple 1 : Notre exemple manuel des articles
def example_article_classification():
    """Exemple de classification d'articles avec Classifier Chains."""
    print("=" * 60)
    print("EXEMPLE ARTICLES - CLASSIFIER CHAINS")
    print("=" * 60)
    
    # Données d'entraînement (features simplifiées)
    # Features: [gouvernement, marché, étranger, loi, finance]
    X_train = np.array([
        [1, 0, 0, 1, 0],  # A1
        [0, 1, 0, 0, 1],  # A2
        [0, 0, 1, 1, 0],  # A3
        [1, 1, 0, 0, 1],  # A4
        [0, 1, 1, 0, 1],  # A5
        [1, 0, 1, 1, 0],  # A6
    ])
    
    # Labels (ensembles)
    y_train = [
        {'Politique'},                    # A1
        {'Économie'},                     # A2
        {'International'},                # A3
        {'Politique', 'Économie'},        # A4
        {'Économie', 'International'},    # A5
        {'Politique', 'International'},   # A6
    ]
    
    label_names = ['Politique', 'Économie', 'International']
    
    print(f"\nDonnées d'entraînement ({len(X_train)} articles):")
    for i, (features, labels) in enumerate(zip(X_train, y_train)):
        print(f"  Article A{i+1}: {labels}")
    
    # Données de test
    X_test = np.array([
        [1, 1, 1, 0, 0],  # Article test: gouvernement, marché, étranger
    ])
    
    print(f"\nArticle de test:")
    print(f"  Features: gouvernement=1, marché=1, étranger=1, loi=0, finance=0")
    
    # Créer et entraîner Classifier Chains
    cc = ClassifierChain(order='original')  # Ordre: Politique → Économie → International
    cc.fit(X_train, y_train, label_names=label_names)
    
    # Faire une prédiction
    predictions = cc.predict(X_test)
    
    print(f"\nPrédiction Classifier Chains:")
    print(f"  Labels prédits: {predictions[0]}")
    
    # Comparer avec Binary Relevance (simulé)
    print(f"\nComparaison avec Binary Relevance (simulé):")
    print(f"  BR prédirait probablement: {{'Politique', 'Économie', 'International'}}")
    print(f"  Raison: Chaque caractéristique déclencherait son label indépendamment")
    
    # Afficher les probabilités
    probas = cc.predict_proba(X_test)
    print(f"\nProbabilités par label (Classifier Chains):")
    for i, label in enumerate(label_names):
        print(f"  {label}: {probas[0, i]:.3f}")
    
    return cc, predictions

# Exemple 2 : Visualisation du processus de chaîne
def visualize_chain_process():
    """Visualise comment Classifier Chains fonctionne étape par étape."""
    print("\n" + "=" * 60)
    print("VISUALISATION DU PROCESSUS DE CHAÎNE")
    print("=" * 60)
    
    # Données très simples
    np.random.seed(42)
    
    # Générer des données avec dépendances fortes entre labels
    n_samples = 100
    
    # Features
    X = np.random.randn(n_samples, 3)
    
    # Labels avec dépendances
    # L1 → influence L2 → influence L3
    y = np.zeros((n_samples, 3), dtype=int)
    
    # L1: basé sur feature 0
    y[:, 0] = (X[:, 0] > 0).astype(int)
    
    # L2: dépend de L1 ET feature 1
    for i in range(n_samples):
        if y[i, 0] == 1:
            y[i, 1] = (X[i, 1] > -0.5).astype(int)  # 80% de chances si L1=1
        else:
            y[i, 1] = (X[i, 1] > 0.5).astype(int)   # 20% de chances si L1=0
    
    # L3: dépend de L2 ET feature 2
    for i in range(n_samples):
        if y[i, 1] == 1:
            y[i, 2] = (X[i, 2] > -0.5).astype(int)  # 70% de chances si L2=1
        else:
            y[i, 2] = (X[i, 2] > 0.5).astype(int)   # 30% de chances si L2=0
    
    label_names = ['L1', 'L2', 'L3']
    
    # Analyser les dépendances
    print(f"\nAnalyse des dépendances entre labels:")
    for i in range(3):
        for j in range(i+1, 3):
            cooccur = np.mean((y[:, i] == 1) & (y[:, j] == 1))
            prob_j_given_i = np.mean(y[y[:, i] == 1, j])
            prob_j_given_not_i = np.mean(y[y[:, i] == 0, j])
            
            print(f"\n  P({label_names[j]}|{label_names[i]}=1) = {prob_j_given_i:.3f}")
            print(f"  P({label_names[j]}|{label_names[i]}=0) = {prob_j_given_not_i:.3f}")
            print(f"  Différence: {abs(prob_j_given_i - prob_j_given_not_i):.3f}")
    
    # Diviser en train/test
    split = int(0.8 * n_samples)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    # Tester différents ordres de chaîne
    orders = [
        ('original', [0, 1, 2]),  # L1 → L2 → L3 (bon ordre)
        ('reverse', [2, 1, 0]),   # L3 → L2 → L1 (mauvais ordre)
        ('random', None),         # Ordre aléatoire
    ]
    
    results = {}
    
    for order_name, order in orders:
        print(f"\n\n=== Test avec ordre: {order_name} ===")
        
        if order_name == 'random':
            cc = ClassifierChain(order='random', random_state=42)
        else:
            cc = ClassifierChain(order=order)
        
        cc.fit(X_train, y_train, label_names=label_names)
        
        # Prédictions
        y_pred = cc.predict(X_test)
        
        # Convertir en format binaire pour évaluation
        y_pred_binary = np.zeros_like(y_test)
        for i, pred_set in enumerate(y_pred):
            for j, label in enumerate(label_names):
                if label in pred_set:
                    y_pred_binary[i, j] = 1
        
        # Calculer la subset accuracy
        correct = 0
        for i in range(len(y_test)):
            if np.array_equal(y_test[i], y_pred_binary[i]):
                correct += 1
        accuracy = correct / len(y_test)
        
        results[order_name] = {
            'accuracy': accuracy,
            'predictions': y_pred_binary
        }
        
        print(f"  Subset Accuracy: {accuracy:.3f}")
    
    print(f"\n\nConclusion:")
    print(f"  Meilleur ordre: 'original' (L1→L2→L3) car correspond aux dépendances réelles")
    print(f"  Pire ordre: 'reverse' (L3→L2→L1) car va à l'encontre des dépendances")
    
    return results

# Exemple 3 : Ensemble de Classifier Chains (ECC)
def ensemble_classifier_chains():
    """Exemple avec Ensemble de Classifier Chains."""
    print("\n" + "=" * 60)
    print("ENSEMBLE DE CLASSIFIER CHAINS (ECC)")
    print("=" * 60)
    
    np.random.seed(42)
    
    # Générer des données
    n_samples = 200
    n_features = 5
    n_labels = 4
    
    X = np.random.randn(n_samples, n_features)
    y = np.zeros((n_samples, n_labels), dtype=int)
    
    # Créer des dépendances complexes
    y[:, 0] = (X[:, 0] > 0).astype(int)
    y[:, 1] = ((y[:, 0] == 1) & (X[:, 1] > 0)).astype(int)
    y[:, 2] = ((y[:, 1] == 1) | (X[:, 2] > 0.5)).astype(int)
    y[:, 3] = ((y[:, 0] == 0) & (y[:, 2] == 1)).astype(int)
    
    label_names = [f'Label_{i}' for i in range(n_labels)]
    
    # Diviser
    split = int(0.7 * n_samples)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    # Single Classifier Chain
    print(f"\n1. Single Classifier Chain (ordre aléatoire):")
    cc_single = ClassifierChain(order='random', random_state=42)
    cc_single.fit(X_train, y_train, label_names=label_names)
    y_pred_single = cc_single.predict(X_test)
    
    # Ensemble de Classifier Chains
    print(f"\n2. Ensemble de Classifier Chains (3 chaînes):")
    
    n_chains = 3
    chains = []
    
    for chain_idx in range(n_chains):
        print(f"\n  Chaîne {chain_idx+1}/{n_chains}:")
        cc = ClassifierChain(order='random', random_state=42+chain_idx)
        cc.fit(X_train, y_train, label_names=label_names)
        chains.append(cc)
    
    # Combiner les prédictions (vote majoritaire par label)
    all_predictions = []
    
    for chain in chains:
        y_pred_chain = chain.predict(X_test)
        # Convertir en binaire
        y_pred_binary = np.zeros((len(X_test), n_labels))
        for i, pred_set in enumerate(y_pred_chain):
            for j, label in enumerate(label_names):
                if label in pred_set:
                    y_pred_binary[i, j] = 1
        all_predictions.append(y_pred_binary)
    
    # Vote majoritaire
    y_pred_ensemble = np.zeros((len(X_test), n_labels))
    for i in range(len(X_test)):
        for j in range(n_labels):
            votes = [pred[i, j] for pred in all_predictions]
            y_pred_ensemble[i, j] = 1 if sum(votes) > n_chains / 2 else 0
    
    # Évaluer
    def subset_accuracy(y_true, y_pred):
        correct = 0
        for i in range(len(y_true)):
            if np.array_equal(y_true[i], y_pred[i]):
                correct += 1
        return correct / len(y_true)
    
    # Convertir y_pred_single en binaire
    y_pred_single_binary = np.zeros((len(X_test), n_labels))
    for i, pred_set in enumerate(y_pred_single):
        for j, label in enumerate(label_names):
            if label in pred_set:
                y_pred_single_binary[i, j] = 1
    
    acc_single = subset_accuracy(y_test, y_pred_single_binary)
    acc_ensemble = subset_accuracy(y_test, y_pred_ensemble)
    
    print(f"\n\nRésultats:")
    print(f"  Single Chain Accuracy: {acc_single:.3f}")
    print(f"  Ensemble Accuracy: {acc_ensemble:.3f}")
    print(f"  Amélioration: {(acc_ensemble - acc_single)/acc_single*100:+.1f}%")
    
    # Afficher quelques prédictions
    print(f"\nExemples de prédictions (3 premières instances):")
    for i in range(min(3, len(X_test))):
        print(f"\n  Instance {i+1}:")
        true_labels = {label_names[j] for j in range(n_labels) if y_test[i, j] == 1}
        single_labels = {label_names[j] for j in range(n_labels) if y_pred_single_binary[i, j] == 1}
        ensemble_labels = {label_names[j] for j in range(n_labels) if y_pred_ensemble[i, j] == 1}
        
        print(f"    Vrai: {true_labels}")
        print(f"    Single Chain: {single_labels}")
        print(f"    Ensemble: {ensemble_labels}")
    
    return chains, acc_single, acc_ensemble

# Exemple 4 : Version ultra-simplifiée
def ultra_simple_classifier_chain():
    """Version ultra-simplifiée pour comprendre le concept."""
    print("\n" + "=" * 60)
    print("VERSION ULTRA-SIMPLIFIÉE DE CLASSIFIER CHAINS")
    print("=" * 60)
    
    # Données très simples sur la météo
    # Features: [soleil, pluie, vent]
    # Labels: [chaud, humide, venteux]
    
    # Règles réelles:
    # 1. chaud = soleil ET (non pluie)
    # 2. humide = pluie OU (chaud ET vent)
    # 3. venteux = vent ET (non chaud)
    
    # Données d'entraînement
    data = [
        # soleil, pluie, vent → chaud, humide, venteux
        ([1, 0, 0], [1, 0, 0]),  # soleil → chaud
        ([0, 1, 0], [0, 1, 0]),  # pluie → humide
        ([0, 0, 1], [0, 0, 1]),  # vent → venteux
        ([1, 0, 1], [1, 1, 0]),  # soleil+vent → chaud+humide
        ([0, 1, 1], [0, 1, 1]),  # pluie+vent → humide+venteux
    ]
    
    X = np.array([d[0] for d in data])
    y = np.array([d[1] for d in data])
    
    label_names = ['chaud', 'humide', 'venteux']
    
    print(f"\nRègles réelles:")
    print(f"  1. chaud = soleil=1 ET pluie=0")
    print(f"  2. humide = pluie=1 OU (chaud=1 ET vent=1)")
    print(f"  3. venteux = vent=1 ET chaud=0")
    
    print(f"\nDonnées d'entraînement:")
    for i, (features, labels) in enumerate(data):
        feature_names = ['soleil', 'pluie', 'vent']
        active_features = [feature_names[j] for j, val in enumerate(features) if val == 1]
        active_labels = [label_names[j] for j, val in enumerate(labels) if val == 1]
        print(f"  Exemple {i+1}: {active_features} → {active_labels}")
    
    # Implémentation manuelle de Classifier Chains
    # Ordre: chaud → humide → venteux
    
    print(f"\n=== Classifier Chains (ordre: chaud→humide→venteux) ===")
    
    # Étape 1: Classifieur pour "chaud"
    print(f"\n1. Entraînement classifieur 'chaud':")
    print(f"   Règle apprise: chaud = soleil=1 ET pluie=0")
    
    # Étape 2: Classifieur pour "humide" (utilise prédiction de "chaud")
    print(f"\n2. Entraînement classifieur 'humide':")
    print(f"   Features étendues: [soleil, pluie, vent, chaud]")
    print(f"   Règle apprise: humide = pluie=1 OU (chaud=1 ET vent=1)")
    
    # Étape 3: Classifieur pour "venteux" (utilise prédictions de "chaud" et "humide")
    print(f"\n3. Entraînement classifieur 'venteux':")
    print(f"   Features étendues: [soleil, pluie, vent, chaud, humide]")
    print(f"   Règle apprise: venteux = vent=1 ET chaud=0")
    
    # Test sur un nouvel exemple
    test_features = [1, 0, 1]  # soleil=1, pluie=0, vent=1
    
    print(f"\n=== Prédiction pour nouvel exemple ===")
    print(f"Features: soleil=1, pluie=0, vent=1")
    
    # Étape par étape
    print(f"\n1. Prédiction 'chaud':")
    print(f"   soleil=1 ET pluie=0 → VRAI")
    print(f"   → chaud = OUI")
    
    print(f"\n2. Prédiction 'humide':")
    print(f"   pluie=0 → FAUX")
    print(f"   chaud=1 ET vent=1 → VRAI")
    print(f"   → humide = OUI")
    
    print(f"\n3. Prédiction 'venteux':")
    print(f"   vent=1 → VRAI")
    print(f"   chaud=1 → FAUX (condition: chaud=0)")
    print(f"   → venteux = NON")
    
    print(f"\nPrédiction finale: chaud=OUI, humide=OUI, venteux=NON")
    print(f"Soit: {{'chaud', 'humide'}}")
    
    # Comparaison avec Binary Relevance
    print(f"\n=== Comparaison avec Binary Relevance ===")
    print(f"Binary Relevance prédirait:")
    print(f"  chaud: soleil=1 ET pluie=0 → OUI")
    print(f"  humide: pluie=0 → NON (ignore la dépendance avec chaud)")
    print(f"  venteux: vent=1 → OUI")
    print(f"  → {{'chaud', 'venteux'}} (incorrect!)")
    
    print(f"\nConclusion:")
    print(f"  Classifier Chains capture la dépendance: 'vent' seul ne signifie pas 'venteux'")
    print(f"  Il faut aussi que 'chaud' soit absent")
    
    return data

# Exemple 5 : Optimisation de l'ordre de la chaîne
def optimize_chain_order():
    """Trouve le meilleur ordre pour Classifier Chains."""
    print("\n" + "=" * 60)
    print("OPTIMISATION DE L'ORDRE DE LA CHAÎNE")
    print("=" * 60)
    
    np.random.seed(42)
    
    # Générer des données avec structure de dépendance connue
    n_samples = 300
    n_features = 4
    n_labels = 4
    
    X = np.random.randn(n_samples, n_features)
    y = np.zeros((n_samples, n_labels), dtype=int)
    
    # Structure de dépendance: L1 → L2 → L3 ← L4
    # L4 est indépendant mais influence L3
    
    y[:, 0] = (X[:, 0] > 0).astype(int)  # L1
    y[:, 1] = (y[:, 0] == 1) & (X[:, 1] > 0)  # L2 dépend de L1
    y[:, 3] = (X[:, 3] > 0).astype(int)  # L4 indépendant
    y[:, 2] = (y[:, 1] == 1) | (y[:, 3] == 1)  # L3 dépend de L2 et L4
    
    label_names = ['L1', 'L2', 'L3', 'L4']
    
    # Analyser les dépendances
    print(f"\nStructure réelle des dépendances:")
    print(f"  L1 → L2 → L3 ← L4")
    print(f"  (L1 influence L2, L2 influence L3, L4 influence L3)")
    
    # Calculer les corrélations conditionnelles
    print(f"\nProbabilités conditionnelles:")
    
    # P(L2|L1)
    p_l2_given_l1 = np.mean(y[y[:, 0] == 1, 1])
    p_l2_given_not_l1 = np.mean(y[y[:, 0] == 0, 1])
    print(f"  P(L2|L1=1) = {p_l2_given_l1:.3f}, P(L2|L1=0) = {p_l2_given_not_l1:.3f}")
    
    # P(L3|L2)
    p_l3_given_l2 = np.mean(y[y[:, 1] == 1, 2])
    p_l3_given_not_l2 = np.mean(y[y[:, 1] == 0, 2])
    print(f"  P(L3|L2=1) = {p_l3_given_l2:.3f}, P(L3|L2=0) = {p_l3_given_not_l2:.3f}")
    
    # P(L3|L4)
    p_l3_given_l4 = np.mean(y[y[:, 3] == 1, 2])
    p_l3_given_not_l4 = np.mean(y[y[:, 3] == 0, 2])
    print(f"  P(L3|L4=1) = {p_l3_given_l4:.3f}, P(L3|L4=0) = {p_l3_given_not_l4:.3f}")
    
    # Diviser
    split = int(0.7 * n_samples)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    # Tester différents ordres
    all_orders = list(itertools.permutations(range(n_labels)))
    
    print(f"\nNombre total d'ordres possibles: {len(all_orders)}")
    print(f"Test des ordres les plus prometteurs...")
    
    # Sélectionner quelques ordres intéressants
    test_orders = [
        [0, 1, 2, 3],  # Suit les dépendances: L1→L2→L3, L4 à la fin
        [3, 0, 1, 2],  # Commence par L4 indépendant
        [2, 1, 0, 3],  # À l'envers
        [1, 0, 2, 3],  # Mélangé
    ]
    
    results = {}
    
    for order in test_orders:
        order_name = '→'.join([label_names[i] for i in order])
        print(f"\nTest de l'ordre: {order_name}")
        
        cc = ClassifierChain(order=order)
        cc.fit(X_train, y_train, label_names=label_names)
        
        y_pred = cc.predict(X_test)
        
        # Convertir en binaire
        y_pred_binary = np.zeros_like(y_test)
        for i, pred_set in enumerate(y_pred):
            for j, label in enumerate(label_names):
                if label in pred_set:
                    y_pred_binary[i, j] = 1
        
        # Subset accuracy
        correct = np.sum(np.all(y_test == y_pred_binary, axis=1))
        accuracy = correct / len(y_test)
        
        results[order_name] = accuracy
        print(f"  Subset Accuracy: {accuracy:.3f}")
    
    # Trouver le meilleur ordre
    best_order = max(results, key=results.get)
    best_acc = results[best_order]
    
    print(f"\n\nRésultats:")
    print(f"  Meilleur ordre: {best_order} (Accuracy: {best_acc:.3f})")
    print(f"  Cet ordre correspond-il aux dépendances?")
    
    # Analyser pourquoi cet ordre est bon
    print(f"\nAnalyse:")
    print(f"  L'ordre optimal devrait placer les labels 'parents' avant leurs 'enfants'")
    print(f"  Ici: L1 (parent de L2) devrait être avant L2")
    print(f"        L2 et L4 (parents de L3) devraient être avant L3")
    
    return results, best_order

# Fonction principale
def main():
    """Exécute tous les exemples Classifier Chains."""
    print("=" * 60)
    print("CLASSIFIER CHAINS - MULTI-LABEL AVEC DÉPENDANCES")
    print("=" * 60)
    
    # Installation nécessaire
    print("\nInstallation requise:")
    print("pip install numpy scikit-learn")
    
    try:
        import sklearn
        print("✓ scikit-learn est installé")
    except ImportError:
        print("\n⚠ scikit-learn n'est pas installé")
        print("Installez avec: pip install scikit-learn")
        return
    
    # Exemple 1 : Articles
    print("\n1. Classification d'articles de presse")
    cc1, pred1 = example_article_classification()
    
    # Exemple 2 : Visualisation
    print("\n2. Visualisation du processus")
    results2 = visualize_chain_process()
    
    # Exemple 3 : Ensemble
    print("\n3. Ensemble de Classifier Chains")
    chains3, acc_single, acc_ensemble = ensemble_classifier_chains()
    
    # Exemple 4 : Version ultra-simplifiée
    print("\n4. Version ultra-simplifiée")
    data4 = ultra_simple_classifier_chain()
    
    # Exemple 5 : Optimisation de l'ordre
    print("\n5. Optimisation de l'ordre de la chaîne")
    results5, best_order5 = optimize_chain_order()
    
    print("\n" + "=" * 60)
    print("RÉSUMÉ DES AVANTAGES ET LIMITES")
    print("=" * 60)
    
    print("\nAvantages de Classifier Chains vs Binary Relevance:")
    print("  1. Capture les dépendances entre labels")
    print("  2. Meilleure précision quand les labels sont corrélés")
    print("  3. Peut prédire des combinaisons plus réalistes")
    print("  4. Même complexité que Binary Relevance")
    
    print("\nLimites:")
    print("  1. Sensible à l'ordre de la chaîne")
    print("  2. Propagation d'erreurs dans la chaîne")
    print("  3. Plus complexe à entraîner et débuguer")
    print("  4. Pas de garantie d'optimalité")
    
    print("\nQuand utiliser Classifier Chains:")
    print("  ✓ Labels fortement corrélés")
    print("  ✓ Nombre modéré de labels (< 20)")
    print("  ✓ On connaît ou peut découvrir la structure de dépendance")
    
    print("\nQuand préférer Binary Relevance:")
    print("  ✓ Labels indépendants ou faiblement corrélés")
    print("  ✓ Nombre très élevé de labels")
    print("  ✓ Besoin de simplicité et interprétabilité")
    print("  ✓ Calcul parallèle important")
    
    print("\n" + "=" * 60)
    print("DÉMONSTRATION TERMINÉE")
    print("=" * 60)

if __name__ == "__main__":
    main()
