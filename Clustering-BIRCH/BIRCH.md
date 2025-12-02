# L'Algorithme BIRCH : Clustering Hiérarchique Équilibré pour Grands Volumes de Données

BIRCH (*Balanced Iterative Reducing and Clustering using Hierarchies*) est un algorithme de clustering conçu pour traiter de très grands jeux de données avec une utilisation mémoire limitée. C'est l'un des plus efficaces pour le clustering à grande échelle.

---

## Pourquoi BIRCH ?

### Limites des algorithmes classiques :

- **k-means** : Nécessite plusieurs passes sur les données, coûteux en I/O.  
- **DBSCAN** : Complexité en *O(n²)*.  
- **DENCLUE** : Calcul de densité sur toutes les données → très coûteux.

### Solution BIRCH :

✔ Une seule passe sur les données  
✔ Construction incrémentale d’un **arbre de résumé** (CF-Tree)

---

## Concept Fondamental : CF (Clustering Feature)

Un **CF** est un triplet statistique qui résume un cluster :

**CF = (N, LS, SS)**

- `N` : nombre de points  
- `LS` : *Linear Sum* (somme des points)  
- `SS` : *Square Sum* (somme des carrés)

### Exemples

#### En 1D
Points : {2, 4, 6}

- N = 3  
- LS = 12  
- SS = 56  

#### En 2D
Points : {(1,2), (3,4)}

- N = 2  
- LS = (4, 6)  
- SS = (10, 20)

---

## Propriétés Magiques des CF

### 1. Additivité
Si CF₁ = (N₁, LS₁, SS₁) et CF₂ = (N₂, LS₂, SS₂) :

**CF₁+₂ = (N₁+N₂, LS₁+LS₂, SS₁+SS₂)**

### 2. Calculs rapides

- **Centroïde** :  
  C = LS / N  

- **Rayon** :  
  R = √(SS/N − (LS/N)²)

- **Diamètre** :  
  D = √(2N × SS − 2 × LS²) / N

---

## Structure de Données : CF-Tree

Le CF-Tree est un arbre **équilibré** (type B+).

### Types de nœuds
- **Nœud feuille** : contient des CF + pointeurs vers données  
- **Nœud interne** : contient des CF résumant ses enfants

### Paramètres importants
- **B** : facteur de branchement (max enfants/nœud interne)  
- **T** : seuil de diamètre/rayon (contrôle compactness)  
- **L** : capacité max de CFs dans un nœud feuille  

---

# Exemple Complet à la Main

## Jeu de données (2D, 10 points)

P1: (1, 1) P2: (1, 2) P3: (2, 1) P4: (2, 2)
P5: (8, 8) P6: (8, 9) P7: (9, 8) P8: (9, 9)
P9: (15, 1) P10: (15, 2)


### Paramètres :

- B = 3  
- T = 2.0  
- L = 3  

---

# Étape par Étape : Construction du CF-Tree

## Phase 1 – Insertion des points

### Point P1(1,1)
- Création d’un CF₁  
- Inséré dans une feuille



Feuille1: [CF₁(N=1, LS=(1,1), SS=(1,1))]


### Point P2, P3, P4
Fusion progressive dans CF₁ (diamètre toujours < T)



CF₁ = (4, (6,6), (10,10))


### Point P5(8,8)
→ nouveau CF₂



Feuille1: [CF₁, CF₂]


### Points P6, P7, P8
Fusion dans CF₂



CF₂ = (4, (34,34), (290,290))


### Point P9
→ nouveau CF₃

### Point P10
Fusion possible avec CF₃ mais **feuille pleine → éclatement**

---

## Split (Éclatement de la feuille)

Distances des CF :

- CF₁ – CF₂ : 9.90  
- CF₁ – CF₃ : 13.51  
- CF₂ – CF₃ : 9.92  

→ Les deux plus éloignés : **CF₁ & CF₃**

Répartition :



Feuille1: [CF₁, CF₂]
Feuille2: [CF₃]


Puis insertion de P10 → fusion avec CF₃

---

# Arbre Final



Racine:
CF_A = (8, (40,40), (300,300))
CF_B = (2, (30,3), (450,5))

Feuille1: [CF₁, CF₂]
Feuille2: [CF₃]


---

# Phase 2 : Clustering Global

On applique un clustering hiérarchique sur les centroïdes :

- CF₁ → (1.5, 1.5)
- CF₂ → (8.5, 8.5)
- CF₃ → (15, 1.5)

Distances :


   CF₁     CF₂     CF₃


CF₁ 0 9.90 13.5
CF₂ 9.90 0 9.92
CF₃ 13.5 9.92 0


Fusion CF₁–CF₂ → puis distance au CF₃ = 10.6 → **2 clusters finaux**

---

# Résultat Final

### Cluster 1 :  
Points P1, P2, P3, P4, P5, P6, P7, P8  

### Cluster 2 :
Points P9, P10

---

# Visualisation (ASCII)



^
| P9 P10
| ● ●
|
| P5 P6
| ● ●
| P7 P8
| ● ●
|
| P3 P4
| ● ●
| P1 P2
| ● ●
+------------------>


---

# Avantages de BIRCH

- Une seule passe sur les données (I/O minimal)  
- Utilisation mémoire contrôlée  
- Robuste au bruit (via T)  
- Incrémentiel : ajoute des données sans tout reconstruire  

---

# Paramètres Clés

- **B** : branching factor  
- **T** : compactness  
  - T petit → beaucoup de petits clusters  
  - T grand → peu de grands  
- **L** : capacité d'une feuille  

---

# Complexité

- **Temps** : O(n)  
- **Mémoire** : O(n/B)  
- **Clustering global** : O(k³), k = nombre de CF feuilles (≪ n)
