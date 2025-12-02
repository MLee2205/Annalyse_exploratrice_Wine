import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, hamming_loss
import matplotlib.pyplot as plt
from collections import defaultdict

class BinaryRelevance:
    """Implémentation simple de Binary Relevance."""
    
    def __init__(self, base_classifier='logistic', **kwargs):
        """
        Parameters:
        -----------
        base_classifier : str, type de classifieur ('logistic', 'naive_bayes', 'custom')
        **kwargs : paramètres pour le classifieur
        """
        self.base_classifier = base_classifier
        self.classifiers = []
        self.label_names = None
        
    def fit(self, X, y, label_names=None):
        """
        Entraîne un classifieur par label.
        
        Parameters:
        -----------
        X : array 2D de forme (n_samples, n_features)
        y : array 2D de forme (n_samples, n_labels) ou list de sets
        label_names : list, noms des labels
        """
        # Convertir y en format binaire si nécessaire
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
        
        # Entraîner un classifieur par label
        self.classifiers = []
        
        for label_idx in range(n_labels):
            print(f"Entraînement du classifieur pour le label '{self.label_names[label_idx]}'...")
            
            # Données pour ce label
            y_label = y_binary[:, label_idx]
            
            # Créer et entraîner le classifieur
            if self.base_classifier == 'logistic':
                clf = LogisticRegression(max_iter=1000, random_state=42, **kwargs)
            elif self.base_classifier == 'naive_bayes':
                clf = MultinomialNB(**kwargs)
            else:
                raise ValueError(f"Classifieur non supporté: {self.base_classifier}")
            
            clf.fit(X, y_label)
            self.classifiers.append(clf)
        
        return self
    
    def predict(self, X):
        """
        Prédit les labels pour chaque instance.
        
        Returns:
        --------
        predictions : list de sets, labels prédits pour chaque instance
        """
        if not self.classifiers:
            raise ValueError("Le modèle doit être entraîné d'abord")
        
        n_samples = X.shape[0]
        n_labels = len(self.classifiers)
        
        # Obtenir les probabilités ou décisions binaires
        predictions = []
        
        for i in range(n_samples):
            instance_pred = set()
            
            for label_idx, clf in enumerate(self.classifiers):
                # Prédire si le label est présent
                pred = clf.predict(X[i:i+1])[0]
                
                if pred == 1:
                    instance_pred.add(self.label_names[label_idx])
            
            predictions.append(instance_pred)
        
        return predictions
    
    def predict_proba(self, X):
        """
        Retourne les probabilités pour chaque label.
        
        Returns:
        --------
        probabilities : array de forme (n_samples, n_labels)
        """
        if not self.classifiers:
            raise ValueError("Le modèle doit être entraîné d'abord")
        
        n_samples = X.shape[0]
        n_labels = len(self.classifiers)
        
        probabilities = np.zeros((n_samples, n_labels))
        
        for label_idx, clf in enumerate(self.classifiers):
            if hasattr(clf, 'predict_proba'):
                proba = clf.predict_proba(X)
                # Probabilité de la classe positive (classe 1)
                probabilities[:, label_idx] = proba[:, 1]
            else:
                # Si pas de predict_proba, utiliser la décision
                probabilities[:, label_idx] = clf.predict(X)
        
        return probabilities
    
    def evaluate(self, X_test, y_test):
        """
        Évalue les performances du modèle.
        
        Returns:
        --------
        metrics : dict, différentes métriques d'évaluation
        """
        y_pred = self.predict(X_test)
        
        # Convertir y_test en format comparable
        if isinstance(y_test[0], set):
            y_test_list = [set(instance) for instance in y_test]
        else:
            y_test_list = []
            for i in range(len(y_test)):
                labels = set()
                for label_idx, val in enumerate(y_test[i]):
                    if val == 1:
                        labels.add(self.label_names[label_idx])
                y_test_list.append(labels)
        
        # Calculer différentes métriques
        metrics = {}
        
        # 1. Exactitude par instance (subset accuracy)
        correct = 0
        for true, pred in zip(y_test_list, y_pred):
            if true == pred:
                correct += 1
        metrics['subset_accuracy'] = correct / len(y_test)
        
        # 2. Hamming Loss (fraction de labels mal prédits)
        total_labels = len(y_test) * len(self.label_names)
        wrong = 0
        
        for true, pred in zip(y_test_list, y_pred):
            # Labels manqués (dans true mais pas dans pred)
            wrong += len(true - pred)
            # Labels extra (dans pred mais pas dans true)
            wrong += len(pred - true)
        
        metrics['hamming_loss'] = wrong / total_labels
        
        # 3. Précision, Rappel, F1 par label
        label_metrics = {}
        
        for label in self.label_names:
            tp = fp = fn = 0
            
            for true, pred in zip(y_test_list, y_pred):
                true_has = label in true
                pred_has = label in pred
                
                if true_has and pred_has:
                    tp += 1
                elif not true_has and pred_has:
                    fp += 1
                elif true_has and not pred_has:
                    fn += 1
            
            # Calcul des métriques
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            label_metrics[label] = {
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'tp': tp,
                'fp': fp,
                'fn': fn
            }
        
        metrics['label_metrics'] = label_metrics
        
        return metrics

# Exemple 1 : Notre exemple manuel des films
def example_movie_classification():
    """Exemple de classification de films avec Binary Relevance."""
    print("=" * 60)
    print("EXEMPLE FILMS - BINARY RELEVANCE")
    print("=" * 60)
    
    # Données d'entraînement (features simplifiées)
    # Vocabulaire: [course, poursuite, explosions, blagues, comique, 
    #               émouvant, famille, espion, combats, humour, drame, relations]
    X_train = np.array([
        [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # F1
        [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0],  # F2
        [0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0],  # F3
        [0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0],  # F4
        [0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1],  # F5
    ])
    
    # Labels (ensembles)
    y_train = [
        {'Action'},           # F1
        {'Comédie'},          # F2
        {'Drame'},            # F3
        {'Action', 'Comédie'}, # F4
        {'Comédie', 'Drame'},  # F5
    ]
    
    label_names = ['Action', 'Comédie', 'Drame']
    
    print(f"\nDonnées d'entraînement ({len(X_train)} films):")
    for i, (features, labels) in enumerate(zip(X_train, y_train)):
        print(f"  Film F{i+1}: {labels}")
    
    # Données de test
    X_test = np.array([
        [1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0],  # Film test: course, comique, émouvant, explosions
    ])
    
    print(f"\nFilm de test:")
    print(f"  Features: course, comique, émouvant, explosions")
    
    # Créer et entraîner le modèle
    br = BinaryRelevance(base_classifier='logistic')
    br.fit(X_train, y_train, label_names=label_names)
    
    # Faire une prédiction
    predictions = br.predict(X_test)
    
    print(f"\nPrédiction pour le film test:")
    print(f"  Labels prédits: {predictions[0]}")
    
    # Afficher les probabilités
    probas = br.predict_proba(X_test)
    print(f"\nProbabilités par label:")
    for i, label in enumerate(label_names):
        print(f"  {label}: {probas[0, i]:.3f}")
    
    return br, predictions

# Exemple 2 : Données synthétiques multi-label
def example_synthetic_data():
    """Exemple avec des données synthétiques."""
    print("\n" + "=" * 60)
    print("DONNÉES SYNTHÉTIQUES MULTI-LABEL")
    print("=" * 60)
    
    np.random.seed(42)
    
    # Générer des données synthétiques
    n_samples = 200
    n_features = 10
    n_labels = 4
    
    # Générer des features
    X = np.random.randn(n_samples, n_features)
    
    # Générer des labels avec certaines corrélations
    y_binary = np.zeros((n_samples, n_labels), dtype=int)
    
    # Label 0: corrélé avec les premières features
    y_binary[:, 0] = (X[:, 0] + X[:, 1] > 0.5).astype(int)
    
    # Label 1: corrélé avec d'autres features
    y_binary[:, 1] = (X[:, 2] - X[:, 3] > 0).astype(int)
    
    # Label 2: combinaison des deux premiers labels
    y_binary[:, 2] = ((y_binary[:, 0] == 1) & (y_binary[:, 1] == 0)).astype(int)
    
    # Label 3: aléatoire
    y_binary[:, 3] = np.random.binomial(1, 0.3, n_samples)
    
    label_names = ['Label_A', 'Label_B', 'Label_C', 'Label_D']
    
    # Statistiques des labels
    print(f"\nStatistiques des labels:")
    for i, name in enumerate(label_names):
        count = np.sum(y_binary[:, i])
        print(f"  {name}: {count} instances ({count/n_samples*100:.1f}%)")
    
    print(f"\nDistribution du nombre de labels par instance:")
    label_counts = np.sum(y_binary, axis=1)
    for count in range(n_labels + 1):
        n_instances = np.sum(label_counts == count)
        print(f"  {count} labels: {n_instances} instances ({n_instances/n_samples*100:.1f}%)")
    
    # Diviser en train/test
    split_idx = int(0.8 * n_samples)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y_binary[:split_idx], y_binary[split_idx:]
    
    # Entraîner Binary Relevance
    br = BinaryRelevance(base_classifier='logistic')
    br.fit(X_train, y_train, label_names=label_names)
    
    # Évaluer
    metrics = br.evaluate(X_test, y_test)
    
    print(f"\nPerformance sur l'ensemble de test:")
    print(f"  Subset Accuracy: {metrics['subset_accuracy']:.3f}")
    print(f"  Hamming Loss: {metrics['hamming_loss']:.3f}")
    
    print(f"\nPerformance par label:")
    for label, label_metrics in metrics['label_metrics'].items():
        print(f"  {label}:")
        print(f"    Precision: {label_metrics['precision']:.3f}")
        print(f"    Rappel: {label_metrics['recall']:.3f}")
        print(f"    F1: {label_metrics['f1']:.3f}")
    
    # Visualiser les prédictions
    y_pred = br.predict(X_test[:10])  # 10 premières instances de test
    
    print(f"\nPrédictions pour 10 premières instances:")
    for i in range(min(10, len(y_pred))):
        true_labels = set()
        for j, val in enumerate(y_test[i]):
            if val == 1:
                true_labels.add(label_names[j])
        
        print(f"  Instance {i}:")
        print(f"    Vrai: {true_labels}")
        print(f"    Prédit: {y_pred[i]}")
        print(f"    Correct: {true_labels == y_pred[i]}")
    
    return br, metrics

# Exemple 3 : Classification de texte multi-label
def example_text_classification():
    """Exemple de classification de texte multi-label."""
    print("\n" + "=" * 60)
    print("CLASSIFICATION DE TEXTE MULTI-LABEL")
    print("=" * 60)
    
    # Documents d'exemple (titres d'articles)
    documents = [
        "Le président annonce des mesures économiques importantes",
        "Match de football intense entre les deux équipes rivales",
        "Nouvelles découvertes scientifiques en médecine génétique",
        "Crise économique et réformes politiques en discussion",
        "Tournoi de tennis avec les meilleurs joueurs mondiaux",
        "Avancées technologiques en intelligence artificielle",
        "Élections présidentielles et débats politiques houleux",
        "Compétition sportive internationale avec records battus",
        "Recherche médicale sur les traitements du cancer",
        "Innovations technologiques dans les énergies renouvelables",
    ]
    
    # Labels pour chaque document (peuvent avoir plusieurs labels)
    y = [
        {'Politique', 'Économie'},
        {'Sport'},
        {'Science', 'Santé'},
        {'Politique', 'Économie'},
        {'Sport'},
        {'Technologie', 'Science'},
        {'Politique'},
        {'Sport'},
        {'Science', 'Santé'},
        {'Technologie', 'Environnement'},
    ]
    
    label_names = ['Politique', 'Économie', 'Sport', 'Science', 'Santé', 'Technologie', 'Environnement']
    
    print(f"\nDocuments ({len(documents)}):")
    for i, (doc, labels) in enumerate(zip(documents, y)):
        print(f"  Doc {i+1}: {doc[:40]}... → {labels}")
    
    # Créer des features simples (bag-of-words)
    from sklearn.feature_extraction.text import CountVectorizer
    
    vectorizer = CountVectorizer(max_features=20)  # Limité à 20 mots pour la simplicité
    X = vectorizer.fit_transform(documents).toarray()
    
    print(f"\nVocabulaire ({len(vectorizer.vocabulary_)} mots):")
    print(f"  {list(vectorizer.get_feature_names_out())}")
    
    # Diviser en train/test
    X_train, X_test = X[:7], X[7:]
    y_train, y_test = y[:7], y[7:]
    
    # Entraîner Binary Relevance
    br = BinaryRelevance(base_classifier='naive_bayes')
    br.fit(X_train, y_train, label_names=label_names)
    
    # Faire des prédictions
    predictions = br.predict(X_test)
    
    print(f"\nPrédictions sur les documents de test:")
    for i, (doc, true_labels, pred_labels) in enumerate(zip(documents[7:], y_test, predictions)):
        print(f"\n  Document {i+8}: '{doc[:40]}...'")
        print(f"    Labels réels: {true_labels}")
        print(f"    Labels prédits: {pred_labels}")
        print(f"    Correct: {true_labels == pred_labels}")
    
    # Évaluer
    metrics = br.evaluate(X_test, y_test)
    
    print(f"\nMétriques d'évaluation:")
    print(f"  Subset Accuracy: {metrics['subset_accuracy']:.3f}")
    print(f"  Hamming Loss: {metrics['hamming_loss']:.3f}")
    
    return br, vectorizer

# Exemple 4 : Version ultra-simplifiée (sans scikit-learn)
def ultra_simple_binary_relevance():
    """Version ultra-simplifiée pour comprendre le concept."""
    print("\n" + "=" * 60)
    print("VERSION ULTRA-SIMPLIFIÉE DE BINARY RELEVANCE")
    print("=" * 60)
    
    # Données très simples
    # Features: [soleil, sable, vague, famille, enfants, aventure]
    # Labels: [plage, famille, vacances]
    
    # Données d'entraînement
    instances = [
        ([1, 1, 1, 0, 0, 0], {'plage'}),           # 1: plage
        ([0, 0, 0, 1, 1, 0], {'famille'}),         # 2: famille
        ([1, 1, 1, 1, 1, 0], {'plage', 'famille'}), # 3: plage et famille
        ([1, 0, 0, 0, 0, 1], {'aventure'}),         # 4: aventure
        ([1, 1, 0, 1, 0, 1], {'plage', 'aventure'}),# 5: plage et aventure
    ]
    
    labels = ['plage', 'famille', 'aventure']
    
    print(f"\nRègles simples pour chaque label:")
    print(f"  - plage: soleil=1 ET (sable=1 OU vague=1)")
    print(f"  - famille: famille=1")
    print(f"  - aventure: aventure=1")
    
    # Classifieurs manuels (règles simples)
    def classifier_plage(features):
        soleil, sable, vague, famille, enfants, aventure = features
        return soleil == 1 and (sable == 1 or vague == 1)
    
    def classifier_famille(features):
        soleil, sable, vague, famille, enfants, aventure = features
        return famille == 1
    
    def classifier_aventure(features):
        soleil, sable, vague, famille, enfants, aventure = features
        return aventure == 1
    
    classifiers = {
        'plage': classifier_plage,
        'famille': classifier_famille,
        'aventure': classifier_aventure
    }
    
    # Tester sur les données d'entraînement
    print(f"\nPerformance sur données d'entraînement:")
    for i, (features, true_labels) in enumerate(instances):
        pred_labels = set()
        
        for label, clf in classifiers.items():
            if clf(features):
                pred_labels.add(label)
        
        correct = true_labels == pred_labels
        print(f"  Instance {i+1}: Vrai={true_labels}, Prédit={pred_labels}, Correct={correct}")
    
    # Tester sur de nouvelles instances
    test_instances = [
        ([1, 1, 0, 1, 1, 0], "plage avec famille"),
        ([1, 0, 1, 0, 0, 1], "plage aventure"),
        ([0, 0, 0, 1, 0, 0], "juste famille"),
    ]
    
    print(f"\nPrédictions sur nouvelles instances:")
    for features, description in test_instances:
        pred_labels = set()
        
        for label, clf in classifiers.items():
            if clf(features):
                pred_labels.add(label)
        
        print(f"  '{description}': {pred_labels}")
    
    return classifiers

# Exemple 5 : Visualisation des performances
def visualize_br_performance():
    """Visualise les performances de Binary Relevance."""
    import matplotlib.pyplot as plt
    
    print("\n" + "=" * 60)
    print("VISUALISATION DES PERFORMANCES")
    print("=" * 60)
    
    # Générer des données
    np.random.seed(42)
    n_samples = 500
    n_features = 15
    n_labels = 5
    
    X = np.random.randn(n_samples, n_features)
    
    # Générer des labels avec différentes difficultés
    y = np.zeros((n_samples, n_labels))
    
    # Labels faciles à prédire
    y[:, 0] = (X[:, 0] > 0).astype(int)  # Très séparable
    y[:, 1] = (X[:, 1] + X[:, 2] > 0.5).astype(int)  # Moyennement séparable
    
    # Labels difficiles
    y[:, 2] = np.random.binomial(1, 0.5, n_samples)  # Aléatoire
    y[:, 3] = (np.sin(X[:, 3] * 5) > 0).astype(int)  # Non linéaire
    y[:, 4] = ((y[:, 0] == 1) & (y[:, 1] == 1)).astype(int)  # Dépendant d'autres labels
    
    label_names = [f'Label_{i}' for i in range(n_labels)]
    
    # Diviser
    split = int(0.7 * n_samples)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    # Entraîner différents classifieurs de base
    results = {}
    
    for classifier_type in ['logistic', 'naive_bayes']:
        print(f"\nEntraînement avec {classifier_type}...")
        br = BinaryRelevance(base_classifier=classifier_type)
        br.fit(X_train, y_train, label_names=label_names)
        
        metrics = br.evaluate(X_test, y_test)
        results[classifier_type] = metrics
    
    # Visualiser
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Métriques globales
    classifiers = list(results.keys())
    subset_acc = [results[clf]['subset_accuracy'] for clf in classifiers]
    hamming_loss_vals = [results[clf]['hamming_loss'] for clf in classifiers]
    
    axes[0, 0].bar(classifiers, subset_acc, color=['blue', 'green'])
    axes[0, 0].set_title('Subset Accuracy')
    axes[0, 0].set_ylabel('Score')
    axes[0, 0].set_ylim(0, 1)
    for i, v in enumerate(subset_acc):
        axes[0, 0].text(i, v + 0.02, f'{v:.3f}', ha='center')
    
    axes[0, 1].bar(classifiers, hamming_loss_vals, color=['red', 'orange'])
    axes[0, 1].set_title('Hamming Loss (plus bas = mieux)')
    axes[0, 1].set_ylabel('Score')
    axes[0, 1].set_ylim(0, 0.5)
    for i, v in enumerate(hamming_loss_vals):
        axes[0, 1].text(i, v + 0.02, f'{v:.3f}', ha='center')
    
    # F1-score par label (pour le premier classifieur)
    clf_type = classifiers[0]
    label_metrics = results[clf_type]['label_metrics']
    
    labels = list(label_metrics.keys())
    f1_scores = [label_metrics[label]['f1'] for label in labels]
    
    axes[1, 0].bar(range(len(labels)), f1_scores, color=plt.cm.tab10(range(len(labels))))
    axes[1, 0].set_title(f'F1-Score par label ({clf_type})')
    axes[1, 0].set_xlabel('Label')
    axes[1, 0].set_ylabel('F1-Score')
    axes[1, 0].set_xticks(range(len(labels)))
    axes[1, 0].set_xticklabels(labels, rotation=45)
    axes[1, 0].set_ylim(0, 1)
    
    # Matrice de confusion des labels (simplifiée)
    # Combien de fois chaque paire de labels apparaît-elle ensemble?
    label_cooccurrence = np.zeros((n_labels, n_labels))
    
    for i in range(n_labels):
        for j in range(n_labels):
            if i != j:
                cooccur = np.sum((y_test[:, i] == 1) & (y_test[:, j] == 1))
                total_i = np.sum(y_test[:, i] == 1)
                if total_i > 0:
                    label_cooccurrence[i, j] = cooccur / total_i
    
    im = axes[1, 1].imshow(label_cooccurrence, cmap='YlOrRd', vmin=0, vmax=1)
    axes[1, 1].set_title('Co-occurrence des labels (test set)')
    axes[1, 1].set_xlabel('Label j')
    axes[1, 1].set_ylabel('Label i')
    axes[1, 1].set_xticks(range(n_labels))
    axes[1, 1].set_yticks(range(n_labels))
    axes[1, 1].set_xticklabels([f'L{i}' for i in range(n_labels)])
    axes[1, 1].set_yticklabels([f'L{i}' for i in range(n_labels)])
    
    plt.colorbar(im, ax=axes[1, 1], label='Proportion')
    
    plt.tight_layout()
    plt.show()
    
    print(f"\nAnalyse:")
    print(f"  Label 0: Facile (linéairement séparable)")
    print(f"  Label 1: Moyen (combinaison linéaire)")
    print(f"  Label 2: Difficile (aléatoire)")
    print(f"  Label 3: Très difficile (non linéaire)")
    print(f"  Label 4: Dépendant (nécessite de capturer les interactions)")
    
    return results

# Fonction principale
def main():
    """Exécute tous les exemples Binary Relevance."""
    print("=" * 60)
    print("BINARY RELEVANCE - CLASSIFICATION MULTI-LABEL")
    print("=" * 60)
    
    # Installation nécessaire
    print("\nInstallation requise:")
    print("pip install numpy matplotlib scikit-learn")
    
    try:
        import sklearn
        print("✓ scikit-learn est installé")
    except ImportError:
        print("\n⚠ scikit-learn n'est pas installé")
        print("Installez avec: pip install scikit-learn")
    
    # Exemple 1 : Films
    print("\n1. Classification de films")
    br1, pred1 = example_movie_classification()
    
    # Exemple 2 : Données synthétiques
    print("\n2. Données synthétiques")
    br2, metrics2 = example_synthetic_data()
    
    # Exemple 3 : Texte
    print("\n3. Classification de texte")
    br3, vectorizer3 = example_text_classification()
    
    # Exemple 4 : Version ultra-simplifiée
    print("\n4. Version ultra-simplifiée")
    classifiers4 = ultra_simple_binary_relevance()
    
    # Exemple 5 : Visualisation
    print("\n5. Visualisation des performances")
    try:
        results5 = visualize_br_performance()
    except Exception as e:
        print(f"  Visualisation échouée: {e}")
    
    print("\n" + "=" * 60)
    print("RÉSUMÉ DES AVANTAGES ET LIMITES")
    print("=" * 60)
    print("\nAvantages de Binary Relevance:")
    print("  1. Simple à comprendre et implémenter")
    print("  2. Évolutif (parallélisable)")
    print("  3. Peut utiliser n'importe quel classifieur binaire")
    print("  4. Pas besoin de modifier les algorithmes existants")
    
    print("\nLimites:")
    print("  1. Ignore les corrélations entre labels")
    print("  2. Peut prédire des combinaisons impossibles")
    print("  3. Nombre de classifieurs = nombre de labels")
    print("  4. Déséquilibre des classes pour certains labels")
    
    print("\n" + "=" * 60)
    print("DÉMONSTRATION TERMINÉE")
    print("=" * 60)

if __name__ == "__main__":
    main()
