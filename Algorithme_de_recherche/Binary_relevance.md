# Binary Relevance (BR) : Classification Multi-Label

Le **Binary Relevance** est une méthode simple mais efficace pour la **classification multi-label**, où chaque instance peut appartenir à plusieurs classes simultanément.

---

## 1. Problème Multi-Label

Différence avec la classification multi-classe :

- **Multi-classe** : Une instance appartient à **UNE SEULE** classe parmi plusieurs  
- **Multi-label** : Une instance peut appartenir à **PLUSIEURS** classes simultanément  

**Exemples :**

- Image : Peut être "plage", "coucher de soleil", "personnes"  
- Document : Peut être "politique", "économie", "international"  
- Musique : Peut être "rock", "années 80", "mélancolique"  

---

## 2. Concept de Binary Relevance

Idée simple : Transformer un problème multi-label en **plusieurs problèmes binaires indépendants**.

Pour M labels possibles :

- Créer M **classifieurs binaires indépendants**  
- Chaque classifieur prédit si un label particulier est présent ou non  
- Combiner les prédictions pour obtenir l'ensemble des labels  

---

## 3. Exemple complet

### Problème : Classification de films

**Labels possibles :**  

- Action (A)  
- Comédie (C)  
- Drame (D)  

**Données d'entraînement (5 films) :**

| Film | Synopsis                    | Labels |
|------|-----------------------------|--------|
| F1   | "Course poursuite explosions" | {A}    |
| F2   | "Blagues situations comiques" | {C}    |
| F3   | "Histoire émouvante famille" | {D}    |
| F4   | "Espion combats humour"       | {A, C} |
| F5   | "Drame comique relations"     | {C, D} |

**Caractéristiques simplifiées (bag-of-words)** :

- **Vocabulaire** : {course, poursuite, explosions, blagues, comique, émouvant, famille, espion, combats, humour, drame, relations}  
- **Représentation binaire (présence/absence)** :

      c  p  e  b  co é  f  es cb h  d  r

F1 = [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0] → A
F2 = [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0] → C
F3 = [0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0] → D
F4 = [0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0] → A, C
F5 = [0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1] → C, D


---

### Étape 1 : Transformer en problèmes binaires

Pour chaque label, créer un jeu de données binaire :

**Label A (Action)**

Film Features Label_A
F1 [1,1,1,0,0,0,0,0,0,0,0] 1
F2 [0,0,0,1,1,0,0,0,0,0,0] 0
F3 [0,0,0,0,0,1,1,0,0,0,0] 0
F4 [0,0,0,1,1,0,0,1,1,1,0] 1
F5 [0,0,0,1,1,1,0,0,0,0,1] 0


**Label C (Comédie)**

Film Features Label_C
F1 [1,1,1,0,0,0,0,0,0,0,0] 0
F2 [0,0,0,1,1,0,0,0,0,0,0] 1
F3 [0,0,0,0,0,1,1,0,0,0,0] 0
F4 [0,0,0,1,1,0,0,1,1,1,0] 1
F5 [0,0,0,1,1,1,0,0,0,0,1] 1


**Label D (Drame)**

Film Features Label_D
F1 [1,1,1,0,0,0,0,0,0,0,0] 0
F2 [0,0,0,1,1,0,0,0,0,0,0] 0
F3 [0,0,0,0,0,1,1,0,0,0,0] 1
F4 [0,0,0,1,1,0,0,1,1,1,0] 0
F5 [0,0,0,1,1,1,0,0,0,0,1] 1


---

### Étape 2 : Entraîner un classifieur par label

Supposons un classifieur simple basé sur **règles logiques** :

- **Classifieur A** : Prédit "Action" si contient ≥2 de {course, poursuite, explosions, espion, combats}  
  - F1 : 3 → OUI  
  - F2 : 0 → NON  
  - F3 : 0 → NON  
  - F4 : 2 → OUI  
  - F5 : 0 → NON  

- **Classifieur C** : Prédit "Comédie" si contient ≥2 de {blagues, comique, humour}  
  - F1 : 0 → NON  
  - F2 : 2 → OUI  
  - F3 : 0 → NON  
  - F4 : 3 → OUI  
  - F5 : 2 → OUI  

- **Classifieur D** : Prédit "Drame" si contient ≥1 de {émouvant, famille, drame, relations}  
  - F1 : 0 → NON  
  - F2 : 0 → NON  
  - F3 : 2 → OUI  
  - F4 : 0 → NON  
  - F5 : ≥1 → OUI  

---

### Étape 3 : Prédiction pour un nouveau film

**Film test** : "Course comique émouvante avec explosions"  
**Features** : course=1, comique=1, émouvant=1, explosions=1, autres=0  
→ `[1,0,1,0,1,1,0,0,0,0,0]`

Application des classifieurs :

- Classifieur A : 2 occurrences → OUI  
- Classifieur C : 1 occurrence → NON  
- Classifieur D : 1 occurrence → OUI  

**Prédiction finale** : `{Action, Drame}
