
========================================
RAPPORT D'EXPLORATION DU DATASET WINE
========================================

# DESCRIPTION DU JEU DE DONNEES WINE

C'est un dataset qui contient les résultats d'une analyse chimique de 178 vins provenant de la même région d'Italie, mais issus de 3 cultivars différents (3 variétés de raisins différentes).

L'objectif est de pouvoir classer automatiquement un vin dans l'une des 3 catégories en se basant sur sa composition chimique.

## COMPOSITION DU DATASET

178 échantillons de vin au total

Classe 1 : 59 vins
Classe 2 : 71 vins
Classe 3 : 48 vins


13 attributs chimiques mesurés pour chaque vin :

Alcool
Acide malique
Cendres
Alcalinité des cendres
Magnésium
Phénols totaux
Flavanoïdes
Phénols non-flavanoïdes
Proanthocyanines
Intensité de la couleur
Teinte
OD280/OD315 des vins dilués
Proline

## PREMIERE ANALYSE VISUELLE DU DATASET

Pas de valeurs manquantes : toutes les données sont complètes

# INTERPRETATION FAITE À PARTIR DES GRAPHIQUES:

## GRAPHE DE REPARTITION DES CLASSES

> Dataset relativement équilibré avec une légère prédominance de la classe 1
> Classe 1: 33.1% | Classe 2: 39.9% | Classe 0: 27.0%
> Pas de déséquilibre majeur qui nécessiterait un rééchantillonnage


## DISTRIBUTION DES CARACTERISTIQUES PAR CLASSE

> Certaines caractéristiques montrent des différences claires entre classes
> Flavanoids et Proline semblent être de bons discriminants
> Quelques caractéristiques ont des distributions qui se chevauchent

## MATRICES DE CORRELATIONS DES CARACTERISTIQUES CHIMIQUES

> Corrélations fortes (>0.7) identifiées:
  - total_phenols & flavanoids: Corrélation:0.865
  - flavanoids & od280/od315_of_diluted_wines: correlation:0.787
> Ces corrélations suggèrent des redondances potentielles
> Pourrait justifier une réduction de dimensionnalité

## BOXPLOTS DES CARACTERISTIQUES PAR CLASSE

---------------------------------------------------------------------------------
 1. flavanoids                | Score: 0.464 | Classe 0 > Classe 1 > Classe 2
 2. proline                   | Score: 0.343 | Classe 0 > Classe 2 > Classe 1
 3. color_intensity           | Score: 0.331 | Classe 2 > Classe 0 > Classe 1
 4. malic_acid                | Score: 0.265 | Classe 2 > Classe 0 > Classe 1
 5. od280/od315_of_diluted_wines | Score: 0.246 | Classe 0 > Classe 1 > Classe 2
 6. total_phenols             | Score: 0.210 | Classe 0 > Classe 1 > Classe 2
 7. proanthocyanins           | Score: 0.198 | Classe 0 > Classe 1 > Classe 2
 8. hue                       | Score: 0.190 | Classe 0 > Classe 1 > Classe 2
 9. nonflavanoid_phenols      | Score: 0.175 | Classe 2 > Classe 1 > Classe 0
10. alcalinity_of_ash         | Score: 0.095 | Classe 2 > Classe 1 > Classe 0

--------------------------------------------------------------------------------

> Meilleures caractéristiques: flavanoids, proline, color_intensity, malic_acid, od280/od315_of_diluted_wines
• Ces caractéristiques montrent des différences nettes entre classes
• Parfaites pour construire un modèle de classification robuste
> Caractéristiques moins discriminantes: magnesium, alcohol, ash
• Ces variables ont des distributions similaires entre classes

## VARIANCE INTERCLASSE

Top 5 des caractéristiques les plus discriminantes:
  1. proline: 67073.677
  2. magnesium: 23.451
  3. alcalinity_of_ash: 3.424
  4. color_intensity: 3.114
  5. flavanoids: 0.816
• Ces caractéristiques devraient être prioritaires pour la classification

NOTE: 1: Calculer la moyenne de chaque classe
      2: Calculer la variance de ces moyennes
      3: Plus la variance est élevée = Mieux c'est !

## ANALYSE PCA

PC1 explique 36.2% de la variance
• PC2 explique 19.2% de la variance
• Les 3 premières composantes expliquent 66.5% de la variance
• Séparation claire des classes dans l'espace PCA
• Réduction de dimensionnalité très efficace possible

NOTE:Étape 1: Standardiser les données (obligatoire!)
Étape 2: Trouver la direction où les données "s'étalent" le plus → PC1
Étape 3: Trouver la 2ème direction (perpendiculaire) → PC2  
Étape 4: Projeter les données sur ces nouvelles directions

# RESUMÉ DES PRINCIPALES DÉCOUVERTES:

1. SÉPARABILITÉ DES CLASSES:
   • Les 3 classes de vins sont bien distinctes chimiquement
   • Certaines caractéristiques (flavanoids, proline) permettent une discrimination excellente
   • Classification automatique très prometteuse

2. CARACTÉRISTIQUES CLÉS:
   • Flavanoids: Meilleur discriminant (Classe 0 >> Classe 1 > Classe 2)
   • Proline: Très élevée pour Classe 0, faible pour Classe 2
   • Color_intensity: Élevée pour Classe 2
   • OD280/OD315: Bon pouvoir discriminant

3. CORRÉLATIONS:
   • Plusieurs caractéristiques fortement corrélées
   • Redondance d'information détectée
   • Réduction de dimensionnalité recommandée

4. ANALYSE PCA:
   • 3 premières composantes capturent la majorité de la variance
   • Excellente séparation des classes dans l'espace réduit
   • Réduction à 5-6 dimensions possible sans perte majeure

# IDENTFIANTS
##Noms 
MEFFO LEA
jecy.meffo@facsciences-uy1.cm
Rapport réalisé dans le cours de INF4117:FOUILLE DE DONNÉES II

