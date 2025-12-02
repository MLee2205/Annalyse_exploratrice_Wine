# Shared Nearest Neighbor (SNN) Similarity Based Clustering

L'algorithme **SNN clustering** est une méthode de clustering basée sur la **similarité des voisins partagés**.  
Contrairement à DBSCAN qui utilise la distance directe, SNN utilise le **nombre de voisins communs** entre les points.

---

## 1. Pourquoi SNN ?

Problèmes avec DBSCAN :

- Sensible au paramètre ε (rayon)  
- Difficulté avec clusters de densités différentes  
- Points dans des régions de densité variable mal traités  

**Solution SNN** : Utilise le recouvrement des k-plus proches voisins.

---

## 2. Concepts fondamentaux

### 2.1 Similarité SNN

La **similarité SNN** entre deux points est le **nombre de voisins qu'ils partagent** dans leurs k-plus proches voisins.

### 2.2 Densité SNN

La **densité SNN** d’un point est le nombre de points ayant une **similarité SNN ≥ seuil** avec lui.

### 2.3 Avantages

- Robuste aux densités variables  
- Détecte mieux les clusters de formes complexes  
- Moins sensible aux paramètres de distance  

---

## 3. Exemple complet

### Données : 8 points en 2D

Points: A(1,1), B(2,2), C(3,3), D(8,8), E(9,9), F(10,10), G(5,5), H(4,6)


---

### Étape 1 : Calculer les k-plus proches voisins (k=3)

**Distances approximatives** :

| Point | Distances aux autres |
|-------|--------------------|
| A     | B:1.4, C:2.8, G:5.7, H:5.8, D:9.9, E:11.3, F:12.7 |
| B     | A:1.4, C:1.4, G:4.2, H:4.5, D:8.5, E:9.9, F:11.3 |
| C     | B:1.4, A:2.8, G:2.8, H:3.6, D:7.1, E:8.5, F:9.9 |
| D     | E:1.4, F:2.8, G:4.2, H:5.7, C:7.1, B:8.5, A:9.9 |
| E     | D:1.4, F:1.4, G:5.7, H:6.4, C:8.5, B:9.9, A:11.3 |
| F     | E:1.4, D:2.8, G:7.1, H:7.2, C:9.9, B:11.3, A:12.7 |
| G     | C:2.8, H:2.2, B:4.2, A:5.7, D:4.2, E:5.7, F:7.1 |
| H     | G:2.2, C:3.6, B:4.5, D:5.7, A:5.8, E:6.4, F:7.2 |

**k=3 plus proches voisins** :

A : {B, C, G}
B : {A, C, G}
C : {B, A, G}
D : {E, F, G}
E : {D, F, G}
F : {E, D, G}
G : {H, C, B}
H : {G, C, B}


---

### Étape 2 : Calculer la similarité SNN

**Matrice SNN (nombre de voisins communs)** :

A B C D E F G H
A - 2 2 0 0 0 2 0
B 2 - 2 0 0 0 1 2
C 2 2 - 0 0 0 1 2
D 0 0 0 - 2 2 0 0
E 0 0 0 2 - 2 0 0
F 0 0 0 2 2 - 0 0
G 2 1 1 0 0 0 - 2
H 0 2 2 0 0 0 2 -


---

### Étape 3 : Définir les liens forts (seuil θ=2)

**Lien fort** : Similarité SNN ≥ θ  

A : {B, C, G}
B : {A, C, H}
C : {A, B, H}
D : {E, F}
E : {D, F}
F : {D, E}
G : {A, H}
H : {B, C, G}


**Matrice d’adjacence des liens forts** :

A B C D E F G H
A - 1 1 0 0 0 1 0
B 1 - 1 0 0 0 0 1
C 1 1 - 0 0 0 0 1
D 0 0 0 - 1 1 0 0
E 0 0 0 1 - 1 0 0
F 0 0 0 1 1 - 0 0
G 1 0 0 0 0 0 - 1
H 0 1 1 0 0 0 1 -


---

### Étape 4 : Construire les clusters

**Composantes connexes** :

- **Cluster 1** : {A, B, C, G, H}  
- **Cluster 2** : {D, E, F}

---

### Étape 5 : Identifier le bruit

**Densité SNN (nombre de liens forts)** :

A:3, B:3, C:3, D:2, E:2, F:2, G:2, H:3


Aucun point n’a densité 0 → Pas de bruit

**Résultat final** :

- Cluster 1 : {A, B, C, G, H}  
- Cluster 2 : {D, E, F}# Shared Nearest Neighbor (SNN) Similarity Based Clustering

L'algorithme **SNN clustering** est une méthode de clustering basée sur la **similarité des voisins partagés**.  
Contrairement à DBSCAN qui utilise la distance directe, SNN utilise le **nombre de voisins communs** entre les points.

---

## 1. Pourquoi SNN ?

Problèmes avec DBSCAN :

- Sensible au paramètre ε (rayon)  
- Difficulté avec clusters de densités différentes  
- Points dans des régions de densité variable mal traités  

**Solution SNN** : Utilise le recouvrement des k-plus proches voisins.

---

## 2. Concepts fondamentaux

### 2.1 Similarité SNN

La **similarité SNN** entre deux points est le **nombre de voisins qu'ils partagent** dans leurs k-plus proches voisins.

### 2.2 Densité SNN

La **densité SNN** d’un point est le nombre de points ayant une **similarité SNN ≥ seuil** avec lui.

### 2.3 Avantages

- Robuste aux densités variables  
- Détecte mieux les clusters de formes complexes  
- Moins sensible aux paramètres de distance  

---

## 3. Exemple complet

### Données : 8 points en 2D

Points: A(1,1), B(2,2), C(3,3), D(8,8), E(9,9), F(10,10), G(5,5), H(4,6)


---

### Étape 1 : Calculer les k-plus proches voisins (k=3)

**Distances approximatives** :

| Point | Distances aux autres |
|-------|--------------------|
| A     | B:1.4, C:2.8, G:5.7, H:5.8, D:9.9, E:11.3, F:12.7 |
| B     | A:1.4, C:1.4, G:4.2, H:4.5, D:8.5, E:9.9, F:11.3 |
| C     | B:1.4, A:2.8, G:2.8, H:3.6, D:7.1, E:8.5, F:9.9 |
| D     | E:1.4, F:2.8, G:4.2, H:5.7, C:7.1, B:8.5, A:9.9 |
| E     | D:1.4, F:1.4, G:5.7, H:6.4, C:8.5, B:9.9, A:11.3 |
| F     | E:1.4, D:2.8, G:7.1, H:7.2, C:9.9, B:11.3, A:12.7 |
| G     | C:2.8, H:2.2, B:4.2, A:5.7, D:4.2, E:5.7, F:7.1 |
| H     | G:2.2, C:3.6, B:4.5, D:5.7, A:5.8, E:6.4, F:7.2 |

**k=3 plus proches voisins** :

A : {B, C, G}
B : {A, C, G}
C : {B, A, G}
D : {E, F, G}
E : {D, F, G}
F : {E, D, G}
G : {H, C, B}
H : {G, C, B}


---

### Étape 2 : Calculer la similarité SNN

**Matrice SNN (nombre de voisins communs)** :

A B C D E F G H
A - 2 2 0 0 0 2 0
B 2 - 2 0 0 0 1 2
C 2 2 - 0 0 0 1 2
D 0 0 0 - 2 2 0 0
E 0 0 0 2 - 2 0 0
F 0 0 0 2 2 - 0 0
G 2 1 1 0 0 0 - 2
H 0 2 2 0 0 0 2 -


---

### Étape 3 : Définir les liens forts (seuil θ=2)

**Lien fort** : Similarité SNN ≥ θ  

A : {B, C, G}
B : {A, C, H}
C : {A, B, H}
D : {E, F}
E : {D, F}
F : {D, E}
G : {A, H}
H : {B, C, G}


**Matrice d’adjacence des liens forts** :

A B C D E F G H
A - 1 1 0 0 0 1 0
B 1 - 1 0 0 0 0 1
C 1 1 - 0 0 0 0 1
D 0 0 0 - 1 1 0 0
E 0 0 0 1 - 1 0 0
F 0 0 0 1 1 - 0 0
G 1 0 0 0 0 0 - 1
H 0 1 1 0 0 0 1 -


---

### Étape 4 : Construire les clusters

**Composantes connexes** :

- **Cluster 1** : {A, B, C, G, H}  
- **Cluster 2** : {D, E, F}

---

### Étape 5 : Identifier le bruit

**Densité SNN (nombre de liens forts)** :

A:3, B:3, C:3, D:2, E:2, F:2, G:2, H:3


Aucun point n’a densité 0 → Pas de bruit

**Résultat final** :

- Cluster 1 : {A, B, C, G, H}  
- Cluster 2 : {D, E, F}
