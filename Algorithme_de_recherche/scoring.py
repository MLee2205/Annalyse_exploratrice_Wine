import numpy as np
import pandas as pd

class SimpleScorecard:
    """Scorecard simple pour évaluation de crédit."""
    
    def __init__(self, base_score=600):
        self.base_score = base_score
        
        # Table de points (basée sur notre analyse manuelle)
        self.point_tables = {
            'age': {
                'bins': [0, 30, 40, 100],
                'labels': ['<30', '30-40', '>40'],
                'points': [-50, 0, 50]
            },
            'income': {
                'bins': [0, 2000, 3000, 100000],
                'labels': ['<2000', '2000-3000', '>3000'],
                'points': [-100, 0, 100]
            },
            'seniority': {
                'bins': [0, 1, 5, 50],
                'labels': ['<1', '1-5', '>5'],
                'points': [-80, 0, 80]
            },
            'debt_ratio': {
                'bins': [0, 0.3, 0.5, 10],
                'labels': ['<0.3', '0.3-0.5', '>0.5'],
                'points': [60, 0, -60]
            }
        }
        
        # Seuils de décision
        self.thresholds = {
            'reject': 550,
            'review': 650,
            'accept': 650
        }
    
    def calculate_score(self, age, income, seniority, debts):
        """Calcule le score pour un client donné."""
        
        # Calcul du ratio dettes/revenu (annuel)
        annual_income = income * 12
        debt_ratio = debts / annual_income if annual_income > 0 else float('inf')
        
        # Initialiser le score
        score = self.base_score
        
        # Points pour l'âge
        age_points = self._get_points('age', age)
        score += age_points
        
        # Points pour le revenu
        income_points = self._get_points('income', income)
        score += income_points
        
        # Points pour l'ancienneté
        seniority_points = self._get_points('seniority', seniority)
        score += seniority_points
        
        # Points pour le ratio dettes/revenu
        debt_ratio_points = self._get_points('debt_ratio', debt_ratio)
        score += debt_ratio_points
        
        return score
    
    def _get_points(self, variable, value):
        """Retourne les points pour une variable et une valeur donnée."""
        table = self.point_tables[variable]
        
        # Trouver dans quel intervalle se trouve la valeur
        for i in range(len(table['bins']) - 1):
            if table['bins'][i] <= value < table['bins'][i + 1]:
                return table['points'][i]
        
        # Si hors limites, retourner le dernier
        return table['points'][-1]
    
    def make_decision(self, score):
        """Prend une décision basée sur le score."""
        if score < self.thresholds['reject']:
            return "REJET", "Score trop bas"
        elif score < self.thresholds['review']:
            return "RÉVISION MANUELLE", "Score limite"
        else:
            return "ACCEPTATION", "Score satisfaisant"
    
    def explain_score(self, age, income, seniority, debts):
        """Donne une explication détaillée du score."""
        print("=" * 60)
        print("EXPLICATION DU CALCUL DU SCORE")
        print("=" * 60)
        
        annual_income = income * 12
        debt_ratio = debts / annual_income if annual_income > 0 else float('inf')
        
        print(f"\nCaractéristiques du client:")
        print(f"  • Âge: {age} ans")
        print(f"  • Revenu mensuel: {income} €")
        print(f"  • Ancienneté emploi: {seniority} ans")
        print(f"  • Dettes existantes: {debts} €")
        print(f"  • Ratio dettes/revenu: {debt_ratio:.2f}")
        
        print(f"\nCalcul détaillé:")
        print(f"  Score de base: {self.base_score} points")
        
        # Points pour chaque variable
        variables = [
            ('age', age, 'Âge'),
            ('income', income, 'Revenu'),
            ('seniority', seniority, 'Ancienneté'),
            ('debt_ratio', debt_ratio, 'Ratio dettes/revenu')
        ]
        
        total_points = self.base_score
        for var_name, value, display_name in variables:
            points = self._get_points(var_name, value)
            total_points += points
            
            cat_info = self._get_category_info(var_name, value)
            print(f"  • {display_name} ({value} → {cat_info}): {points:+d} points")
        
        print(f"\nScore total: {total_points} points")
        
        # Décision
        decision, reason = self.make_decision(total_points)
        print(f"\nDécision: {decision}")
        print(f"Raison: {reason}")
        
        return total_points

    def _get_category_info(self, variable, value):
        """Retourne la catégorie d'une variable."""
        table = self.point_tables[variable]
        
        for i in range(len(table['bins']) - 1):
            if table['bins'][i] <= value < table['bins'][i + 1]:
                return table['labels'][i]
        
        return table['labels'][-1]

# Exemple d'utilisation
def example_manual_scoring():
    """Exemple de scoring manuel."""
    
    print("=" * 60)
    print("EXEMPLE DE SCORING DE CRÉDIT")
    print("=" * 60)
    
    # Créer notre scorecard
    scorecard = SimpleScorecard(base_score=600)
    
    # Test avec nos clients historiques
    clients = [
        # (nom, âge, revenu, ancienneté, dettes, défaut réel)
        ("C1 (mauvais)", 25, 2000, 1, 5000, 1),
        ("C2 (bon)", 35, 3500, 5, 10000, 0),
        ("C3 (bon)", 42, 2800, 3, 8000, 0),
        ("C4 (mauvais)", 28, 2200, 2, 15000, 1),
        ("C5 (bon)", 50, 5000, 10, 20000, 0),
        ("C6 (mauvais)", 31, 1800, 1, 12000, 1),
        ("C7 (bon)", 45, 4000, 8, 5000, 0),
        ("C8 (très mauvais)", 26, 1500, 0.5, 8000, 1),
        ("C9 (bon)", 38, 3200, 4, 6000, 0),
        ("C10 (bon limite)", 29, 2500, 2, 10000, 0)
    ]
    
    print("\nÉVALUATION DES CLIENTS HISTORIQUES")
    print("-" * 60)
    
    results = []
    for name, age, income, seniority, debts, default in clients:
        score = scorecard.calculate_score(age, income, seniority, debts)
        decision, reason = scorecard.make_decision(score)
        
        results.append({
            'client': name,
            'score': score,
            'decision': decision,
            'default_real': "Défaut" if default else "Bon",
            'correct': (decision == "REJET" and default == 1) or 
                      (decision != "REJET" and default == 0)
        })
        
        print(f"\n{name}:")
        print(f"  Score: {score}")
        print(f"  Décision: {decision}")
        print(f"  Réalité: {'Défaut' if default else 'Bon payeur'}")
    
    # Statistiques
    print("\n" + "=" * 60)
    print("PERFORMANCE DU MODÈLE")
    print("=" * 60)
    
    n_correct = sum(1 for r in results if r['correct'])
    n_total = len(results)
    accuracy = n_correct / n_total
    
    print(f"\nPrécision globale: {n_correct}/{n_total} = {accuracy:.1%}")
    
    # Détail par type
    print("\nDétail des décisions:")
    df_results = pd.DataFrame(results)
    print(df_results[['client', 'score', 'decision', 'default_real']].to_string(index=False))
    
    return scorecard, results

# Version 2 : Scoring avec régression logistique
class LogisticScoringModel:
    """Modèle de scoring basé sur la régression logistique."""
    
    def __init__(self):
        self.coefficients = None
        self.intercept = None
        self.feature_names = None
        
    def fit(self, X, y, feature_names=None):
        """Entraîne le modèle sur des données historiques."""
        from sklearn.linear_model import LogisticRegression
        
        # Entraîner une régression logistique
        model = LogisticRegression()
        model.fit(X, y)
        
        # Stocker les coefficients
        self.coefficients = model.coef_[0]
        self.intercept = model.intercept_[0]
        self.feature_names = feature_names if feature_names else [f"Feature_{i}" for i in range(X.shape[1])]
        
        # Convertir en points (méthode standard)
        self._create_scorecard_from_logit()
        
        return self
    
    def _create_scorecard_from_logit(self):
        """Convertit les coefficients logistiques en points de score."""
        # Facteur d'échelle standard: 20 points pour doublement des odds
        self.pdo = 20  # Points to Double the Odds
        self.base_odds = 1.0  # Odds de base
        self.base_score = 600  # Score de base
        
        # Calcul du facteur et offset
        self.factor = self.pdo / np.log(2)
        self.offset = self.base_score - self.factor * np.log(self.base_odds)
        
    def predict_score(self, X):
        """Prédit le score pour de nouvelles données."""
        if self.coefficients is None:
            raise ValueError("Le modèle doit être entraîné d'abord")
        
        # Calculer le score linéaire
        scores = []
        for sample in X:
            linear_score = self.intercept
            for i, value in enumerate(sample):
                linear_score += self.coefficients[i] * value
            
            # Convertir en probabilité puis en score
            probability = 1 / (1 + np.exp(-linear_score))
            odds = probability / (1 - probability) if probability < 1 else float('inf')
            
            # Convertir odds en score
            if odds > 0:
                score = self.offset + self.factor * np.log(odds)
            else:
                score = 0
            
            scores.append(score)
        
        return np.array(scores)
    
    def predict_proba(self, X):
        """Retourne la probabilité de défaut."""
        if self.coefficients is None:
            raise ValueError("Le modèle doit être entraîné d'abord")
        
        linear_scores = np.dot(X, self.coefficients) + self.intercept
        probabilities = 1 / (1 + np.exp(-linear_scores))
        
        return probabilities
    
    def display_coefficients(self):
        """Affiche les coefficients avec leur importance."""
        print("\nCOEFFICIENTS DU MODÈLE:")
        print("-" * 50)
        
        for i, (name, coef) in enumerate(zip(self.feature_names, self.coefficients)):
            importance = abs(coef) / np.sum(np.abs(self.coefficients))
            direction = "↑ risque" if coef > 0 else "↓ risque"
            print(f"{name:15} : {coef:7.3f} ({importance:5.1%}) {direction}")
        
        print(f"\nIntercept (constante) : {self.intercept:.3f}")

# Exemple avec données simulées
def example_logistic_scoring():
    """Exemple de scoring avec régression logistique."""
    
    print("\n" + "=" * 60)
    print("SCORING AVEC RÉGRESSION LOGISTIQUE")
    print("=" * 60)
    
    # Générer des données simulées
    np.random.seed(42)
    n_samples = 1000
    
    # Variables explicatives
    age = np.random.normal(35, 10, n_samples)
    income = np.random.normal(3000, 1000, n_samples)
    seniority = np.random.exponential(3, n_samples)
    
    # Ratio dettes/revenu (corrélé négativement avec le revenu)
    debt_ratio = 0.5 - 0.0001 * income + np.random.normal(0, 0.2, n_samples)
    debt_ratio = np.maximum(debt_ratio, 0.05)  # Minimum 5%
    
    # Créer la variable cible (défaut)
    # Probabilité de défaut basée sur les variables
    log_odds = (-0.05 * age +  # Âge plus élevé = moins de risque
                -0.0005 * income +  # Revenu plus élevé = moins de risque
                -0.1 * seniority +  # Ancienneté plus élevée = moins de risque
                2.0 * debt_ratio +  # Ratio élevé = plus de risque
                np.random.normal(0, 1, n_samples))
    
    probability_default = 1 / (1 + np.exp(-log_odds))
    default = (probability_default > 0.5).astype(int)
    
    # Afficher les statistiques
    print(f"\nDonnées générées:")
    print(f"  • Nombre d'échantillons: {n_samples}")
    print(f"  • Taux de défaut: {default.mean():.1%}")
    print(f"  • Âge moyen: {age.mean():.1f} ans")
    print(f"  • Revenu moyen: {income.mean():.1f} €")
    
    # Préparer les données
    X = np.column_stack([age, income, seniority, debt_ratio])
    y = default
    
    # Créer et entraîner le modèle
    feature_names = ['Âge', 'Revenu', 'Ancienneté', 'Ratio dettes/revenu']
    model = LogisticScoringModel()
    model.fit(X, y, feature_names=feature_names)
    
    # Afficher les coefficients
    model.display_coefficients()
    
    # Tester sur quelques exemples
    print("\n" + "=" * 50)
    print("TESTS SUR NOUVEAUX CLIENTS")
    print("=" * 50)
    
    test_cases = [
        (25, 2000, 1, 0.4, "Jeune, faible revenu"),
        (45, 5000, 10, 0.2, "Âgé, bon revenu"),
        (30, 3000, 2, 0.6, "Ratio élevé")
    ]
    
    for age_val, income_val, seniority_val, ratio_val, description in test_cases:
        X_test = np.array([[age_val, income_val, seniority_val, ratio_val]])
        
        # Probabilité de défaut
        proba = model.predict_proba(X_test)[0]
        
        # Score
        score = model.predict_score(X_test)[0]
        
        print(f"\n{description}:")
        print(f"  • Âge: {age_val} ans, Revenu: {income_val} €")
        print(f"  • Ancienneté: {seniority_val} ans, Ratio: {ratio_val:.1%}")
        print(f"  • Probabilité de défaut: {proba:.1%}")
        print(f"  • Score: {score:.0f}")
        
        if score < 550:
            decision = "REJET"
        elif score < 650:
            decision = "RÉVISION"
        else:
            decision = "ACCEPTATION"
        
        print(f"  • Décision: {decision}")
    
    return model, X, y

# Version 3 : Scorecard interactive
class InteractiveScorecard:
    """Scorecard interactive pour démonstration."""
    
    def __init__(self):
        self.scorecard = SimpleScorecard()
        
    def run_interactive(self):
        """Lance une interface interactive."""
        print("\n" + "=" * 60)
        print("SCORECARD INTERACTIVE - SIMULATION DE PRÊT")
        print("=" * 60)
        print("\nEntrez les informations du client:")
        
        # Saisie interactive
        age = float(input("Âge (ans): "))
        income = float(input("Revenu mensuel net (€): "))
        seniority = float(input("Ancienneté dans l'emploi actuel (ans): "))
        debts = float(input("Montant total des dettes existantes (€): "))
        
        # Montant demandé
        loan_amount = float(input("Montant du prêt demandé (€): "))
        loan_duration = float(input("Durée du prêt (ans): "))
        
        # Calcul détaillé
        print("\n" + "=" * 60)
        score = self.scorecard.explain_score(age, income, seniority, debts)
        
        # Analyse supplémentaire
        monthly_payment = loan_amount / (loan_duration * 12)
        debt_ratio = debts / (income * 12)
        
        print("\n" + "=" * 60)
        print("ANALYSE COMPLÉMENTAIRE")
        print("=" * 60)
        
        print(f"\n1. CAPACITÉ DE REMBOURSEMENT:")
        print(f"   • Mensualité proposée: {monthly_payment:.2f} €/mois")
        print(f"   • Taux d'endettement: {(monthly_payment / income * 100):.1f}% du revenu")
        
        if monthly_payment > income * 0.33:
            print("   ⚠  DANGER: Mensualité trop élevée (>33% du revenu)")
        elif monthly_payment > income * 0.25:
            print("   ⚠  ATTENTION: Mensualité élevée (25-33% du revenu)")
        else:
            print("   ✓ OK: Mensualité raisonnable (<25% du revenu)")
        
        print(f"\n2. SITUATION FINANCIÈRE:")
        print(f"   • Ratio dettes/revenu actuel: {debt_ratio:.2f}")
        
        if debt_ratio > 0.5:
            print("   ⚠  DANGER: Niveau d'endettement très élevé")
        elif debt_ratio > 0.3:
            print("   ⚠  ATTENTION: Niveau d'endettement élevé")
        else:
            print("   ✓ OK: Niveau d'endettement acceptable")
        
        print(f"\n3. RECOMMANDATION FINALE:")
        decision, reason = self.scorecard.make_decision(score)
        
        if decision == "REJET":
            print(f"   ❌ {decision}: {reason}")
            print(f"   Suggestions:")
            print(f"   • Réduire le montant du prêt demandé")
            print(f"   • Allonger la durée du prêt")
            print(f"   • Attendre d'avoir plus d'ancienneté")
        elif decision == "RÉVISION MANUELLE":
            print(f"   ⚠  {decision}: {reason}")
            print(f"   Documents requis:")
            print(f"   • 3 dernières fiches de paie")
            print(f"   • Avis d'imposition")
            print(f"   • Relevés bancaires des 3 derniers mois")
        else:
            print(f"   ✅ {decision}: {reason}")
            print(f"   Conditions proposées:")
            print(f"   • Taux: {(3 + max(0, (650 - score)/100)):.1f}%")
            print(f"   • Assurance: {(0.2 + max(0, (650 - score)/500)):.1f}%")
        
        return score, decision

# Exemple 4 : Évaluation de performance
def evaluate_scoring_performance():
    """Évalue la performance d'un modèle de scoring."""
    
    print("\n" + "=" * 60)
    print("ÉVALUATION DE PERFORMANCE DU SCORING")
    print("=" * 60)
    
    # Générer des données de test
    np.random.seed(123)
    n_samples = 500
    
    # Variables
    scores = np.random.normal(600, 100, n_samples)
    true_default = (scores < 550) + np.random.binomial(1, 0.2, n_samples) * (scores >= 550)
    true_default = true_default > 0
    
    # Seuils à tester
    thresholds = np.arange(400, 800, 25)
    
    results = []
    
    for threshold in thresholds:
        # Prédictions
        predicted_reject = scores < threshold
        
        # Métriques
        tp = np.sum(predicted_reject & true_default)  # Vrais positifs (correctly rejected bad)
        tn = np.sum(~predicted_reject & ~true_default)  # Vrais négatifs (correctly accepted good)
        fp = np.sum(predicted_reject & ~true_default)  # Faux positifs (wrongly rejected good)
        fn = np.sum(~predicted_reject & true_default)  # Faux négatifs (wrongly accepted bad)
        
        if tp + fp > 0:
            precision = tp / (tp + fp)
        else:
            precision = 0
            
        if tp + fn > 0:
            recall = tp / (tp + fn)
        else:
            recall = 0
            
        accuracy = (tp + tn) / n_samples
        
        results.append({
            'threshold': threshold,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn
        })
    
    # Trouver le meilleur seuil
    best_result = max(results, key=lambda x: x['accuracy'])
    
    print(f"\nAnalyse de {n_samples} prêts simulés")
    print(f"Meilleur seuil: {best_result['threshold']:.0f} points")
    print(f"Précision: {best_result['accuracy']:.1%}")
    print(f"Précision (parmi rejets): {best_result['precision']:.1%}")
    print(f"Rappel (défauts détectés): {best_result['recall']:.1%}")
    
    print(f"\nMatrice de confusion (seuil = {best_result['threshold']}):")
    print(f"                    | Prédit BON | Prédit MAUVAIS |")
    print(f"--------------------|------------|----------------|")
    print(f"Réel BON           |   {best_result['tn']:4d}      |     {best_result['fp']:4d}       |")
    print(f"Réel MAUVAIS       |   {best_result['fn']:4d}      |     {best_result['tp']:4d}       |")
    
    # Courbe ROC simplifiée
    #print(f"\nCourbe ROC (simplifiée):")
    """
    # Tracer une courbe ROC simple
    import matplotlib.pyplot as plt
    
    tpr = [r['recall'] for r in results]
    fpr = [r['fp'] / (r['fp'] + r['tn']) if (r['fp'] + r['tn']) > 0 else 0 for r in results]
    
    plt.figure(figsize=(10, 6))
    plt.plot(fpr, tpr, 'b-', linewidth=2)
    plt.plot([0, 1], [0, 1], 'r--', alpha=0.5)
    plt.scatter(fpr[thresholds == best_result['threshold']][0], 
                tpr[thresholds == best_result['threshold']][0], 
                color='red', s=100, zorder=5)
    plt.xlabel('Taux de Faux Positifs (bons clients rejetés)')
    plt.ylabel('Taux de Vrais Positifs (mauvais détectés)')
    plt.title('Courbe ROC du Modèle de Scoring')
    plt.grid(True, alpha=0.3)
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.show()
    """
    return results, best_result

# Fonction principale
def main():
    """Exécute tous les exemples de scoring."""
    print("=" * 60)
    print("SCORING - CRÉATION DE SCORES PRÉDICTIFS")
    print("=" * 60)
    
    # Exemple 1 : Scorecard manuelle
    print("\n1. SCORECARD MANUELLE (NOTRE EXEMPLE)")
    scorecard, results = example_manual_scoring()
    
    # Exemple 2 : Scoring avec régression logistique
    print("\n2. SCORING STATISTIQUE (RÉGRESSION LOGISTIQUE)")
    model, X, y = example_logistic_scoring()
    
    # Exemple 3 : Scorecard interactive
    print("\n3. SCORECARD INTERACTIVE")
    interactive = InteractiveScorecard()
    # Décommentez pour essayer interactivement:
    # score_interactive, decision = interactive.run_interactive()
    
    # Exemple 4 : Évaluation de performance
    print("\n4. ÉVALUATION DE PERFORMANCE")
    all_results, best_result = evaluate_scoring_performance()
    
    print("\n" + "=" * 60)
    print("CONCLUSION SUR LE SCORING")
    print("=" * 60)
    print("""
    Le scoring transforme des caractéristiques complexes en un 
    nombre simple pour prendre des décisions.
    
    Étapes clés:
    1. Analyse des données historiques
    2. Discrétisation des variables continues
    3. Attribution de points par catégorie
    4. Définition des seuils de décision
    5. Validation et calibration
    
    Applications:
    • Crédit bancaire (score FICO, etc.)
    • Assurance (score de risque)
    • Marketing (score de propension à acheter)
    • RH (score de recrutement)
    
    Métriques importantes:
    • Précision : % de décisions correctes
    • Recall : % de mauvais clients détectés
    • Spécificité : % de bons clients acceptés
    • AUC-ROC : Performance globale du modèle
    """)

if __name__ == "__main__":
    main()
