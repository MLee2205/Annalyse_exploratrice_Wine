# Le Ranking (Classement)

Le **ranking** est le processus d'ordonnancement d'items selon un critère de pertinence ou de qualité.  
Contrairement au scoring qui donne une valeur absolue, le ranking donne une **position relative**.

---

# 1. Concepts Fondamentaux

## 1.1 Types de ranking
- **Explicite** : Basé sur des notes/évaluations (films, produits)
- **Implicite** : Basé sur le comportement (clics, temps passé)
- **Hybride** : Combinaison des deux

## 1.2 Mesures de qualité
- **NDCG** (Normalized Discounted Cumulative Gain)
- **MAP** (Mean Average Precision)
- **MRR** (Mean Reciprocal Rank)
- **Kendall Tau** (corrélation de rangs)

---

# 2. EXEMPLE À LA MAIN COMPLET : Classement de produits

**Scénario :** Site e-commerce avec 5 produits à classer

Données disponibles :
- Note moyenne (sur 5)
- Nombre d'avis (fiabilité)
- Pertinence de recherche (0-1)
- Taux de conversion (%)

---

# Étape 1 : Données des produits

| Produit | Note | Nombre avis | Pertinence | Conversion |
|---------|------|-------------|------------|------------|
| A | 4.8 | 1200 | 0.9 | 5.2% |
| B | 4.5 | 50 | 0.7 | 3.1% |
| C | 4.9 | 20 | 0.8 | 2.8% |
| D | 4.2 | 300 | 0.6 | 4.5% |
| E | 3.9 | 500 | 0.5 | 6.0% |

**Problème :** Comment classer ces produits ?

---

# Étape 2 : Méthode 1 — Score simple

Approche : créer un score combiné

**Formule :**  
Score = (Note × 0.4) + (log(Avis) × 0.2) + (Pertinence × 0.3) + (Conversion × 0.1)

## Calculs

### Produit A
- Note: 4.8 × 0.4 = 1.92  
- log(1200) ≈ 7.09 × 0.2 = 1.418  
- Pertinence: 0.9 × 0.3 = 0.27  
- Conversion: 5.2 × 0.1 = 0.52  
**Total = 4.128**

### Produit B
Total = 1.8 + 0.782 + 0.21 + 0.31 = **3.102**

### Produit C
Total = 1.96 + 0.6 + 0.24 + 0.28 = **3.08**

### Produit D
Total = 1.68 + 1.14 + 0.18 + 0.45 = **3.45**

### Produit E
Total = 1.56 + 1.242 + 0.15 + 0.6 = **3.552**

### Classement par score
1. **A : 4.128**  
2. **E : 3.552**  
3. **D : 3.45**  
4. **B : 3.102**  
5. **C : 3.08**

---

# Étape 3 : Méthode 2 — Méthode de Borda

Principe : chaque produit reçoit des points selon sa position dans chaque critère.  
Pour **5 produits** : 1er=4 pts, 2e=3 pts, 3e=2 pts, 4e=1 pt, 5e=0 pt.

## Étape 3.1 : Classement par critère

### Par Note
C > A > B > D > E

### Par Nombre d'avis
A > E > D > B > C

### Par Pertinence
A > C > B > D > E

### Par Conversion
E > A > D > B > C

## Étape 3.2 : Points Borda

| Produit | Note | Avis | Pertinence | Conversion | Total |
|---------|------|------|------------|------------|--------|
| A | 3 | 4 | 4 | 3 | **14** |
| B | 2 | 1 | 2 | 1 | **6** |
| C | 4 | 0 | 3 | 0 | **7** |
| D | 1 | 2 | 1 | 2 | **6** |
| E | 0 | 3 | 0 | 4 | **7** |

### Classement Borda
1. **A : 14 points**  
2. **E : 7 points** (ex aequo)  
3. **C : 7 points** (ex aequo)  
4. **B : 6 points**  
5. **D : 6 points**

---

# Étape 4 : Méthode 3 — PageRank simplifié

Principe : les produits se "votent" entre eux selon leur similarité.

## Matrice de similarité
- A → C, D  
- B → D  
- C → A  
- D → A, E  
- E → D

## Matrice de transition normalisée

A B C D E
A 0 0 0.5 0.5 0
B 0 0 0 1 0
C 1 0 0 0 0
D 0.5 0 0 0 0.5
E 0 0 0 1 0


## Initialisation : PR = [0.2, 0.2, 0.2, 0.2, 0.2]

### Résultats après itération (α=0.85)

- **PR(D) = 0.455**
- **PR(A) = 0.285**
- **PR(C) = 0.115**
- **PR(E) = 0.115**
- **PR(B) = 0.03**

### Classement PageRank
1. **D**
2. **A**
3. **C**
4. **E**
5. **B**

---

# Étape 5 : Comparaison des méthodes

| Produit | Score simple | Borda | PageRank | Moyenne rang |
|----------|--------------|--------|-----------|----------------|
| A | 1 | 1 | 2 | **1.33** |
| B | 4 | 4.5 | 5 | **4.5** |
| C | 5 | 2.5 | 3.5 | **3.67** |
| D | 3 | 4.5 | 1 | **2.83** |
| E | 2 | 2.5 | 3.5 | **2.67** |

---

# Classement final consensuel

1. **A** — Bon partout  
2. **E** — Meilleure conversion  
3. **D** — Bon équilibre  
4. **C** — Bonne note mais peu d'avis  
5. **B** — Faible partout
