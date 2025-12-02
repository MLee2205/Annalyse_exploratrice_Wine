# Classifier Chains (CC) : Chaînes de Classifieurs pour le Multi-Label 🔗

Le **Classifier Chains** est une méthode avancée pour la classification **multi-label** qui prend en compte les **dépendances entre labels** contrairement à Binary Relevance.

---

## 1. Problème avec Binary Relevance

**Binary Relevance** : Traite chaque label indépendamment

* Ignore que certains labels apparaissent souvent ensemble
* Peut prédire des combinaisons impossibles
* Exemple : "**océan**" et "**désert**" sont rarement ensemble

**Solution Classifier Chains** : Ordonner les labels en chaîne, chaque classifieur utilise les prédictions précédentes.

---

## 2. Concept de Classifier Chains

**Idée clé** :
* **Ordre des labels** : $L_1 \rightarrow L_2 \rightarrow L_3 \rightarrow \dots \rightarrow L_m$
* **Classifieur pour $L_i$ utilise :**
    * Les features originales $\mathbf{X}$
    * Les prédictions des labels précédents $L_1$ à $L_{i-1}$

**Avantage** :
* Capture les **dépendances conditionnelles** entre labels
    * *Si $L_1$ est présent, $L_2$ a 80% de chances de l'être aussi.*
    * *Si $L_1$ est absent, $L_2$ a seulement 10% de chances.*

---

## 3. EXEMPLE À LA MAIN COMPLET

**Problème** : Classification d'articles de presse

**Labels possibles** :
* **P**olitique (P)
* **É**conomie (E)
* **I**nternational (I)

**Caractéristiques** : [gouvernement, marché, étranger, loi, finance]

### Données d'entraînement (6 articles) :

| Article | Caractéristiques | Labels |
|:-------:|:---------------:|:------:|
| A1 | [1,0,0,1,0] | {P} |
| A2 | [0,1,0,0,1] | {E} |
| A3 | [0,0,1,1,0] | {I} |
| A4 | [1,1,0,0,1] | {P, E} |
| A5 | [0,1,1,0,1] | {E, I} |
| A6 | [1,0,1,1,0] | {P, I} |

### Étape 1 : Définir l'ordre de la chaîne

**Ordre choisi : Politique $\rightarrow$ Économie $\rightarrow$ International**

### Étape 2 : Préparer les données pour chaque classifieur

#### Classifieur 1 : Politique (P)
*Utilise seulement les caractéristiques originales.*

| Article | Features | $Label_P$ |
|:-------:|:--------:|:---------:|
| A1 | [1,0,0,1,0] | **1** |
| A4 | [1,1,0,0,1] | **1** |
| A6 | [1,0,1,1,0] | **1** |
| A2, A3, A5 | [..] | **0** |

#### Classifieur 2 : Économie (E)
*Utilise : Features + prédiction de Politique*

| Article | Features | $Préd\_P$ | $Label_E$ |
|:-------:|:--------:|:---------:|:---------:|
| A1 | [1,0,0,1,0] | **1** | 0 |
| A4 | [1,1,0,0,1] | **1** | **1** |
| A2 | [0,1,0,0,1] | 0 | **1** |
| A5 | [0,1,1,0,1] | 0 | **1** |

#### Classifieur 3 : International (I)
*Utilise : Features + prédictions de P et E*

| Article | Features | $Préd\_P$ | $Préd\_E$ | $Label_I$ |
|:-------:|:--------:|:---------:|:---------:|:---------:|
| A3 | [0,0,1,1,0] | 0 | 0 | **1** |
| A5 | [0,1,1,0,1] | 0 | **1** | **1** |
| A6 | [1,0,1,1,0] | **1** | 0 | **1** |
| A1, A2, A4 | [..] | [..] | [..] | 0 |

### Étape 3 : Construire des règles simples (pour l'exemple)

* **Politique (P)** : Présent si "**gouvernement=1**" OU "**loi=1**"
* **Économie (E)** : Présent si "**marché=1**" OU (**"finance=1" ET P=0**)
* **International (I)** : Présent si "**étranger=1**" ET (**P=0 OU E=0**)
    * *Remarque : La règle pour I utilise explicitement les prédictions précédentes (P et E).*

### Étape 4 : Prédiction pour un nouvel article

**Article test :** "Le gouvernement discute des marchés étrangers"
**Features** : [gouvernement=1, marché=1, étranger=1, loi=0, finance=0] $\rightarrow \mathbf{X} = [1, 1, 1, 0, 0]$

1.  **Politique (P)** :
    * Règle : gouvernement=1 $\checkmark$
    * $\rightarrow \mathbf{P=1}$
    * Features étendues pour le suivant : $\mathbf{X}_{ext\_2} = [1, 1, 1, 0, 0, \mathbf{1}]$

2.  **Économie (E)** :
    * Règle : marché=1 $\checkmark$ (ou (finance=1 ET P=0))
    * $\rightarrow \mathbf{E=1}$
    * Features étendues pour le suivant : $\mathbf{X}_{ext\_3} = [1, 1, 1, 0, 0, 1, \mathbf{1}]$

3.  **International (I)** :
    * Règle : étranger=1 $\checkmark$ ET (**P=0 OU E=0**)?
        * $P=1$ (Faux)
        * $E=1$ (Faux)
        * Condition (P=0 OU E=0) = **FAUX**
    * $\rightarrow \mathbf{I=0}$

**Prédiction finale : {Politique, Économie}**

### Étape 5 : Comparaison avec Binary Relevance

Si on utilisait Binary Relevance (indépendance) :
* **Politique** : gouvernement=1 $\checkmark \rightarrow$ OUI
* **Économie** : marché=1 $\checkmark \rightarrow$ OUI
* **International** : étranger=1 $\checkmark \rightarrow$ OUI

* **Binary Relevance prédirait** : {P, E, I} (trop optimiste)
* **Classifier Chains prédit** : {P, E} (plus réaliste car l'absence de la condition $P=0$ ou $E=0$ bloque $I$).
