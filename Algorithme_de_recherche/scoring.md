# EXEMPLE À LA MAIN COMPLET : Score de Crédit  
**Scénario :** Banque qui évalue des demandes de prêt

Variables disponibles :

- Âge (ans)
- Revenu mensuel (€)
- Ancienneté emploi (ans)
- Dettes existantes (€)

---

# Étape 1 : Analyse des données historiques

Base de données de **10 clients** :

| Client | Âge | Revenu | Ancienneté | Dettes | Défaut |
|--------|-----|--------|------------|--------|---------|
| C1 | 25 | 2000 | 1 | 5000 | 1 |
| C2 | 35 | 3500 | 5 | 10000 | 0 |
| C3 | 42 | 2800 | 3 | 8000 | 0 |
| C4 | 28 | 2200 | 2 | 15000 | 1 |
| C5 | 50 | 5000 | 10 | 20000 | 0 |
| C6 | 31 | 1800 | 1 | 12000 | 1 |
| C7 | 45 | 4000 | 8 | 5000 | 0 |
| C8 | 26 | 1500 | 0.5 | 8000 | 1 |
| C9 | 38 | 3200 | 4 | 6000 | 0 |
| C10 | 29 | 2500 | 2 | 10000 | 0 |

---

# Étape 2 : Discrétisation des variables

## Âge
- **< 30 ans : Risque élevé**  
  (C1, C4, C6, C8, C10 → 4 défauts / 5)
- **30–40 ans : Risque moyen**  
  (C2, C9 → 0 défauts)
- **> 40 ans : Risque faible**  
  (C3, C5, C7 → 0 défauts)

## Revenu mensuel
- **< 2000 € : Risque élevé**  
  (C8 → 1/1)
- **2000–3000 € : Risque moyen**  
  (C1, C3, C4, C6, C10 → 3/5)
- **> 3000 € : Risque faible**  
  (C2, C5, C7, C9 → 0/4)

## Ancienneté emploi
- **< 1 an : Risque élevé**  
  (C1, C6, C8 → 3/3)
- **1–5 ans : Risque moyen**  
  (C3, C4, C9, C10 → 1/4)
- **> 5 ans : Risque faible**  
  (C2, C5, C7 → 0/3)

## Dettes existantes
- **< 8000 € : Risque faible**  
  (C1, C3, C7, C9 → 1/4)
- **8000–15000 € : Risque moyen**  
  (C2, C4, C6, C8, C10 → 3/5)
- **> 15000 € : Risque élevé**  
  (C5 → 0/1) → mais variable peu discriminante ici.

---

# Étape 3 : Création de la table de points

**Principe :** plus le risque est faible, plus on donne de points.  
Base de score : **600 points**

| Variable | Catégorie | Points | Logique |
|----------|-----------|--------|---------|
| Âge | < 30 ans | -50 | Risque élevé |
|      | 30–40 ans | 0 | Moyen |
|      | > 40 ans | +50 | Faible |
| Revenu | < 2000 € | -100 | Élevé |
|         | 2000–3000 € | 0 | Moyen |
|         | > 3000 € | +100 | Faible |
| Ancienneté | < 1 an | -80 | Élevé |
|             | 1–5 ans | 0 | Moyen |
|             | > 5 ans | +80 | Faible |
| Dettes/Revenu | > 5 | -60 | Endettement élevé |
|               | 3–5 | 0 | Moyen |
|               | < 3 | +60 | Faible |

**Ratio Dettes/Revenu = Dettes annuelles / Revenu annuel**

Exemple :  
C1 → 5000 / (2000 × 12) = 5000 / 24000 ≈ 0.21 (faible)

---

# Étape 4 : Calcul des scores

**Formule :**  
Score = 600 + points(Âge) + points(Revenu) + points(Ancienneté) + points(Ratio)

---

## Client C1
- Âge 25 → -50  
- Revenu 2000 → 0  
- Ancienneté 1 an → -80  
- Ratio < 3 → +60  

**Score = 600 - 50 - 80 + 60 = 530**

---

## Client C2
- Âge 35 → 0  
- Revenu 3500 → +100  
- Ancienneté 5 ans → 0  
- Ratio < 3 → +60  

**Score = 600 + 0 + 100 + 0 + 60 = 760**

---

## Client C8
- Âge 26 → -50  
- Revenu 1500 → -100  
- Ancienneté 0.5 → -80  
- Ratio < 3 → +60  

**Score = 600 - 50 - 100 - 80 + 60 = 430**

---

# Étape 5 : Seuils de décision

| Client | Score | Défaut réel | Décision |
|--------|--------|-------------|-----------|
| C1 | 530 | 1 | Rejet |
| C2 | 760 | 0 | Acceptation |
| C3 | 690 | 0 | Acceptation |
| C4 | 570 | 1 | Rejet |
| C5 | 810 | 0 | Acceptation |
| C6 | 490 | 1 | Rejet |
| C7 | 790 | 0 | Acceptation |
| C8 | 430 | 1 | Rejet |
| C9 | 720 | 0 | Acceptation |
| C10 | 610 | 0 | Acceptation |

**Distribution :**
- Mauvais payeurs : 430, 490, 530, 570 (moyenne 505)
- Bons payeurs : 610, 690, 720, 760, 790, 810 (moyenne 730)

**Seuils proposés :**
- Score < 550 → Rejet automatique  
- Score 550–650 → Révision manuelle  
- Score > 650 → Acceptation automatique

---

# Étape 6 : Test sur un nouveau client

Nouvelle demande :

- Âge : 32 ans  
- Revenu : 2800 €  
- Ancienneté : 2 ans  
- Dettes : 15000 €

**Calcul :**
- Âge 32 → 0  
- Revenu 2800 → 0  
- Ancienneté 2 ans → 0  
- Ratio = 15000 / (2800×12) ≈ 0.45 → +60  

**Score final = 600 + 60 = 660**

**Décision : Score 660 > 650 → PRÊT ACCEPTÉ**

