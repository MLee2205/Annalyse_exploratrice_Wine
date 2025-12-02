# EXEMPLE À LA MAIN COMPLET (Ondelettes de Haar)

## Signal d'exemple

[1, 2, 3, 4, 5, 6, 7, 8]


---

# Étape 1 : Comprendre l'ondelette de Haar

### Fonction d'échelle (passe-bas) : Moyenne  
h = [1/√2, 1/√2]  
ou simplifié (non normalisé) :  

[1, 1]


### Fonction ondelette (passe-haut) : Différence  
g = [1/√2, -1/√2]  
ou simplifié :  

[1, -1]


---

# Étape 2 : Niveau 1 de décomposition

### Signal :  

S = [1, 2, 3, 4, 5, 6, 7, 8]


## Calcul des approximations (A1)
On prend des paires → moyenne :

- (1, 2) → 1.5  
- (3, 4) → 3.5  
- (5, 6) → 5.5  
- (7, 8) → 7.5  

**A1 = [1.5, 3.5, 5.5, 7.5]**

## Calcul des détails (D1)
Différence entre les paires :

- (1, 2) → -0.5  
- (3, 4) → -0.5  
- (5, 6) → -0.5  
- (7, 8) → -0.5  

**D1 = [-0.5, -0.5, -0.5, -0.5]**

---

# Étape 3 : Niveau 2 de décomposition

Travail sur :

A1 = [1.5, 3.5, 5.5, 7.5]


## Approximations (A2)

- (1.5, 3.5) → 2.5  
- (5.5, 7.5) → 6.5  

**A2 = [2.5, 6.5]**

## Détails (D2)

- (1.5, 3.5) → -1.0  
- (5.5, 7.5) → -1.0  

**D2 = [-1.0, -1.0]**

---

# Étape 4 : Niveau 3 de décomposition

Travail sur :

A2 = [2.5, 6.5]


## Approximation (A3)
- (2.5, 6.5) → 4.5  

**A3 = [4.5]**

## Détail (D3)
- (2.5, 6.5) → -2.0  

**D3 = [-2.0]**

---

# Étape 5 : Résultat final de la DWT

## Coefficients DWT (format standard)

[A3, D3, D2, D1]
= [4.5, -2.0, -1.0, -1.0, -0.5, -0.5, -0.5, -0.5]


## Structure arborescente

Niveau 3 : A3 = [4.5], D3 = [-2.0]
Niveau 2 : D2 = [-1.0, -1.0]
Niveau 1 : D1 = [-0.5, -0.5, -0.5, -0.5]


---

# Étape 6 : Reconstruction (IDWT — Inverse DWT)

## Niveau 3 → Niveau 2
À partir de A3 = [4.5] et D3 = [-2.0] :

- 4.5 + (-2.0) = 2.5  
- 4.5 - (-2.0) = 6.5  

**A2 reconstruit = [2.5, 6.5] ✓**

---

## Niveau 2 → Niveau 1
À partir de A2 et D2 :

- 2.5 + (-1.0) = 1.5  
- 2.5 - (-1.0) = 3.5  
- 6.5 + (-1.0) = 5.5  
- 6.5 - (-1.0) = 7.5  

**A1 reconstruit = [1.5, 3.5, 5.5, 7.5] ✓**

---

## Niveau 1 → Signal original
À partir de A1 et D1 :

- 1.5 ± (-0.5) → 1, 2  
- 3.5 ± (-0.5) → 3, 4  
- 5.5 ± (-0.5) → 5, 6  
- 7.5 ± (-0.5) → 7, 8  

**Signal reconstruit = [1, 2, 3, 4, 5, 6, 7, 8] ✓ Parfait !**

