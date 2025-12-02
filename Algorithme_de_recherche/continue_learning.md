# Le Continual Learning (Apprentissage Continu)

Le **Continual Learning** (ou *Lifelong Learning*) est la capacité d'un modèle à apprendre continuellement de nouvelles tâches **sans oublier les anciennes**.  
C’est un défi majeur pour créer des IA apprenant comme les humains.

---

## 1. Le Problème : L'Oubli Catastrophique

**Oubli catastrophique** : lorsqu'un modèle apprend une nouvelle tâche, il oublie complètement les précédentes.

**Exemple :**

- Étape 1 : apprentissage des *chats* → **95%** de précision  
- Étape 2 : apprentissage des *chiens*  
- Résultat : performance sur *chats* → **10%** (oubli total)

---

## 2. Exemple complet à la main

### Scénario : Reconnaissance de chiffres MNIST simplifié

Trois tâches :

- **Tâche 1** : reconnaître {0, 1}
- **Tâche 2** : reconnaître {2, 3}
- **Tâche 3** : reconnaître {4, 5}

---

## Étape 1 : Modèle simple (sans continual learning)

**Architecture :**

- Entrée : 3 neurones  
- Caché : 4 neurones  
- Sortie : 2 neurones  

### Poids initiaux

W1 (entrée→caché) = [[0.1, 0.2, 0.3],
[0.4, 0.5, 0.6],
[0.7, 0.8, 0.9],
[1.0, 1.1, 1.2]]

W2 (caché→sortie) = [[0.2, 0.3, 0.4, 0.5],
[0.6, 0.7, 0.8, 0.9]]


---

### Tâche 1 : apprentissage de {0,1}

Données :

- 0 → `[0,0,1]` → `[1,0]`
- 1 → `[0,1,0]` → `[0,1]`

Poids après entraînement T1 :

W1_après_T1 = [[0.15, 0.25, 0.35],
[0.45, 0.55, 0.65],
[0.75, 0.85, 0.95],
[1.05, 1.15, 1.25]]

W2_après_T1 = [[0.25, 0.35, 0.45, 0.55],
[0.65, 0.75, 0.85, 0.95]]


**Performance T1 : 98%**

---

### Tâche 2 : apprentissage {2, 3}  
➡️ PROBLÈME : on réinitialise la sortie → oubli total de T1

Données :

- 2 → `[1,0,0]` → `[1,0]`
- 3 → `[1,1,0]` → `[0,1]`

Poids après T2 (standard) :

W1_après_T2 = [[0.05, 0.15, 0.25],
[0.35, 0.45, 0.55],
[0.65, 0.75, 0.85],
[0.95, 1.05, 1.15]]

W2_après_T2 = [[0.15, 0.25, 0.35, 0.45],
[0.55, 0.65, 0.75, 0.85]]


Résultat sur Tâche 1 : **10%** → **Oubli catastrophique !**

---

## Étape 2 : Solution 1 — Elastic Weight Consolidation (EWC)

**Idée** : Protéger les poids importants pour les anciennes tâches.

Importance des poids (simplifiée) :

F_W1 = [[0.8, 0.7, 0.6],
[0.5, 0.4, 0.3],
[0.2, 0.1, 0.05],
[0.01, 0.005, 0.001]]

F_W2 = [[0.9, 0.8, 0.7, 0.6],
[0.5, 0.4, 0.3, 0.2]]


Formule EWC :

Loss_total = Loss_T2 + λ × Σ( Fᵢ × (θᵢ - θ*ᵢ)² )


Poids après T2 avec EWC :

W1_EWC = [[0.14, 0.24, 0.34],
[0.44, 0.54, 0.64],
[0.60, 0.70, 0.80],
[0.90, 1.00, 1.10]]

W2_EWC = [[0.24, 0.34, 0.44, 0.54],
[0.60, 0.70, 0.80, 0.90]]


Performances :

- Tâche 1 : **95%**  
- Tâche 2 : **92%**

---

## Étape 3 : Solution 2 — Replay Memory

**Idée :** conserver des exemples des anciennes tâches.

Méthode :

- Pendant T2 :
  - 90% des batches = données T2
  - 10% = exemples mémorisés de T1

Avantage : pas d’oubli catastrophique  
Limite : mémoire limitée

---

## Étape 4 : Solution 3 — Architecture dynamique

**Idée :** ajouter des neurones à chaque nouvelle tâche.

Exemple :

- Après T1 → 4 neurones cachés
- Après T2 → ajouter 2 neurones → 6 cachés

Avantage : poids T1 préservés  
Limite : croissance infinie du réseau

---

