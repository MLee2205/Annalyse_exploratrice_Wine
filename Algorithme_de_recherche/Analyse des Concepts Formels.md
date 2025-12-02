# Analyse des Concepts Formels (Formal Concept Analysis - FCA)

L’**Analyse des Concepts Formels** (FCA) est une méthode mathématique permettant d’extraire des concepts et une hiérarchie à partir de **données binaires**.  
Elle transforme une table d’incidence en **treillis de concepts**.

---

## 1. Concepts fondamentaux

### 1.1 Contexte formel

Un contexte est un triplet **(G, M, I)** où :

- **G** : ensemble d’objets  
- **M** : ensemble d’attributs  
- **I ⊆ G × M** : relation d’incidence (un objet possède un attribut)

---

### 1.2 Opérateurs de dérivation

Pour **A ⊆ G** (ensemble d’objets) :

A' = { m ∈ M | ∀ g ∈ A : (g, m) ∈ I }


→ attributs communs aux objets de A

Pour **B ⊆ M** (ensemble d’attributs) :

B' = { g ∈ G | ∀ m ∈ B : (g, m) ∈ I }


→ objets possédant tous les attributs de B

---

### 1.3 Concept formel

Un **concept** est une paire (A, B) telle que :

- A ⊆ G = **extension**
- B ⊆ M = **intension**

Avec :

- A' = B  
- B' = A

---

## 2. Exemple complet

### Contexte : animaux et caractéristiques

#### Objets (G)

- Chat (C)
- Chien (D)
- Oiseau (O)
- Poisson (P)
- Serpent (S)

#### Attributs (M)

- M : Mammifère  
- V : Volant  
- N : Nageant  
- R : Carnivore  
- D : Domestique  

#### Table d’incidence

M V N R D
C 1 0 0 1 1
D 1 0 1 1 1
O 0 1 0 0 0
P 0 0 1 0 0
S 0 0 0 1 0


---

## Étape 1 : Calcul des concepts

### Concept 1 : concept suprême (TOP)

- Extension : {C, D, O, P, S}  
- Intension : ∅  

Concept : **({C,D,O,P,S}, ∅)**

---

### Concept 2 : carnivores

- Intension : {R}  
- Extension : R' = {C, D, S}

Concept : **({C,D,S}, {R})**

---

### Concept 3 : mammifères

- Intension candidate : {M}
- Extension : M' = {C, D}
- Attributs communs : {M, R, D}

Concept : **({C,D}, {M,R,D})**

---

### Concept 4 : nageants

- Intension : {N}
- Extension : {D, P}

Concept : **({D,P}, {N})**

---

### Concept 5 : volants

- Intension : {V}
- Extension : {O}

Concept : **({O}, {V})**

---

### Concept 6–10 : combinaisons d’attributs

Toutes les paires `{M,D}`, `{M,R}`, `{R,D}`, `{M,R,D}`  
donnent **le même concept que C3**.

---

### Concept 11 : concept infimum (BOTTOM)

- Intension : {M,V,N,R,D}
- Extension : ∅

Concept : **(∅, {M,V,N,R,D})**

---

## Étape 2 : Vérification de fermeture A = A''

Exemples :

- {C} → pas fermé  
- {O} → fermé → Concept 5  
- {D,P} → fermé → Concept 4  
- {C,D} → fermé → Concept 3  
- {C,D,S} → fermé → Concept 2  

---

## Étape 3 : Treillis des concepts

Concepts retenus :

- C1 : ({C,D,O,P,S}, ∅)
- C2 : ({C,D,S}, {R})
- C3 : ({C,D}, {M,R,D})
- C4 : ({D,P}, {N})
- C5 : ({O}, {V})
- C11 : (∅, {M,V,N,R,D})

Relations :

- C3 ⊆ C2  
- C3 ⊆ C4  
- C2, C4, C5 ⊆ C1  
- C11 ⊆ tous  

Treillis :

      C1 ({C,D,O,P,S}, ∅)
     /      |       \
  C2       C4       C5

({C,D,S},{R}) ({D,P},{N}) ({O},{V})
\ | /
\ | /
C3 ({C,D},{M,R,D})
|
|
C11 (∅,{M,V,N,R,D})


---

## Étape 4 : Interprétation

Concepts clés :

- **C3** : mammifères carnivores domestiques = {Chat, Chien}  
- **C2** : carnivores = {Chat, Chien, Serpent}  
- **C4** : nageants = {Chien, Poisson}  
- **C5** : volants = {Oiseau}  

Insights :

- Tous les mammifères sont carnivores **et** domestiques  
- Le chien est le seul mammifère nageant  
- Aucun attribut n’est commun à tous les animaux  
- Le serpent partage seulement “carnivore”  

---
