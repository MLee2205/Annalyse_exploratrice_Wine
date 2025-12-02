# 1. Qu'est-ce qu'un Motif Graduel ?

Un **motif graduel** est un ensemble d'attributs qui varient simultanément de manière cohérente dans un jeu de données.

Contrairement aux règles d'association classiques (A → B), les motifs graduels capturent des **tendances évolutives** :

- **Augmentation simultanée** :  
  *"Quand les prix des maisons augmentent, les taux d'intérêt augmentent aussi"*

- **Diminution simultanée** :  
  *"Quand la température baisse, la consommation de chauffage baisse"*

- **Variations inverses** :  
  *"Quand le prix augmente, la demande diminue"*

---

# 2. Définitions Mathématiques

## 2.1 Base de données graduelle

Soit une base de données **D** avec :

- **n** objets (lignes)
- **m** attributs (colonnes)
- Chaque cellule **v(i,j)** est une valeur numérique

## 2.2 Variation graduelle

Pour un attribut **A** et deux objets **o₁, o₂** :

- **A⁺** : o₁[A] < o₂[A] (augmentation)  
- **A⁻** : o₁[A] > o₂[A] (diminution)

## 2.3 Support d'un motif graduel

Le **support** d'un motif graduel **P** est la proportion de paires d'objets *(oᵢ, oⱼ)* où tous les attributs de **P** varient dans la direction indiquée.

---

# 3. Exemple complet à la main  
## Base de données : Ventes de produits

| Client | Dépenses (€) | Fréquence visites | Satisfaction (1–10) |
|--------|--------------|-------------------|----------------------|
| C1     | 100          | 2                 | 8                    |
| C2     | 150          | 4                 | 9                    |
| C3     | 80           | 1                 | 6                    |
| C4     | 200          | 5                 | 9                    |
| C5     | 120          | 3                 | 7                    |

Objectif : **Trouver un motif graduel comme  
"Quand les dépenses augmentent, la satisfaction augmente aussi"**

---

# Étape 1 : Calculer toutes les paires d’objets

Nous avons 5 clients, donc **C(5,2) = 10 paires** :

(C1,C2), (C1,C3), (C1,C4), (C1,C5),  
(C2,C3), (C2,C4), (C2,C5),  
(C3,C4), (C3,C5), (C4,C5)

---

# Étape 2 : Motif {Dépenses⁺, Satisfaction⁺}

Condition :  
Dépenses(Ci) < Dépenses(Cj) **ET** Satisfaction(Ci) < Satisfaction(Cj)

### Paires valides :

- **(C1,C2)** → ✓ ✓ → **Valide**  
- (C1,C3) → ✗  
- **(C1,C4)** → ✓ ✓ → **Valide**  
- (C1,C5) → ✗  
- (C2,C3) → ✗  
- (C2,C4) → ✗  
- (C2,C5) → ✗  
- **(C3,C4)** → ✓ ✓ → **Valide**  
- **(C3,C5)** → ✓ ✓ → **Valide**  
- (C4,C5) → ✗  

**4 paires valides sur 10 → Support = 0.4 (40%)**

---

# Étape 3 : Motif {Dépenses⁺, Fréquence⁺}

Paires valides :

(C1,C2), (C1,C4), (C1,C5),  
(C2,C4), (C3,C4), (C3,C5)

**6 paires valides → Support = 6/10 = 0.6 (60%)**

---

# Étape 4 : Motif {Dépenses⁺, Satisfaction⁺, Fréquence⁺}

Paires valides :

(C1,C2), (C1,C4), (C3,C4), (C3,C5)

**Support = 4/10 = 0.4 (40%)**

---

# Étape 5 : Interprétation

Motifs graduels (seuil = 30%) :

- **{Dépenses⁺, Fréquence⁺} — Support = 60%**  
  → *"Quand les clients dépensent plus, ils visitent plus souvent"*

- **{Dépenses⁺, Satisfaction⁺} — Support = 40%**  
  → *"Quand les dépenses augmentent, la satisfaction augmente aussi"*

- **{Dépenses⁺, Satisfaction⁺, Fréquence⁺} — Support = 40%**  
  → *"Quand les dépenses augmentent, satisfaction ET fréquence augmentent"*

