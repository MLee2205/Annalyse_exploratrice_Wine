## Bi-clustering

## Algorithme Cheng & Church (Exemple à la Main)

C'est l'un des algorithmes les plus connus. Il cherche des biclusters avec variance résiduelle faible.
Notion clé : Score de Variance Résiduelle (Mean Squared Residue - MSR)

Pour un bicluster (I,J) :

    Calculer la moyenne de chaque ligne : a_iJ

    Calculer la moyenne de chaque colonne : a_Ij

    Calculer la moyenne globale : μ

    Score MSR = 1/(|I||J|) × Σ (a_ij - a_iJ - a_Ij + μ)²

Un bicluster parfait a MSR = 0.
Matrice de Données (4 clients × 5 produits)

Notes de satisfaction (1-10) :
text

      P1  P2  P3  P4  P5
C1     5   6   4   7   8
C2     6   7   5   8   9
C3     9   8  10   7   6
C4     8   7   9   6   5

Étape 1 : Recherche d'un premier bicluster

Objectif : Trouver un sous-ensemble de clients et produits avec un pattern cohérent.
Essai 1 : Regardons les colonnes P1, P2, P3
text

      P1  P2  P3
C1     5   6   4
C2     6   7   5
C3     9   8  10
C4     8   7   9

Calcul du MSR pour le sous-matrice complète (4×3) :

    Moyennes par ligne :

        C1: (5+6+4)/3 = 5.00

        C2: (6+7+5)/3 = 6.00

        C3: (9+8+10)/3 = 9.00

        C4: (8+7+9)/3 = 8.00

    Moyennes par colonne :

        P1: (5+6+9+8)/4 = 7.00

        P2: (6+7+8+7)/4 = 7.00

        P3: (4+5+10+9)/4 = 7.00

    Moyenne globale :
    μ = (5+6+4+6+7+5+9+8+10+8+7+9)/12 = 84/12 = 7.00

    Calcul des résidus :
    Résidu r_ij = a_ij - a_iJ - a_Ij + μ

    Pour C1-P1 : 5 - 5.00 - 7.00 + 7.00 = 0.00
    Pour C1-P2 : 6 - 5.00 - 7.00 + 7.00 = 1.00
    Pour C1-P3 : 4 - 5.00 - 7.00 + 7.00 = -1.00

    Pour C2-P1 : 6 - 6.00 - 7.00 + 7.00 = 0.00
    Pour C2-P2 : 7 - 6.00 - 7.00 + 7.00 = 1.00
    Pour C2-P3 : 5 - 6.00 - 7.00 + 7.00 = -1.00

    Pour C3-P1 : 9 - 9.00 - 7.00 + 7.00 = 0.00
    Pour C3-P2 : 8 - 9.00 - 7.00 + 7.00 = -1.00
    Pour C3-P3 : 10 - 9.00 - 7.00 + 7.00 = 1.00

    Pour C4-P1 : 8 - 8.00 - 7.00 + 7.00 = 0.00
    Pour C4-P2 : 7 - 8.00 - 7.00 + 7.00 = -1.00
    Pour C4-P3 : 9 - 8.00 - 7.00 + 7.00 = 1.00

    MSR = (0²+1²+(-1)²+0²+1²+(-1)²+0²+(-1)²+1²+0²+(-1)²+1²)/12
    = (0+1+1+0+1+1+0+1+1+0+1+1)/12 = 8/12 = 0.667

MSR = 0.667 (pas terrible, nous voulons un score plus proche de 0)
Essai 2 : Examinons seulement C1 et C2 avec P1, P2, P3
text

      P1  P2  P3
C1     5   6   4
C2     6   7   5

Calcul du MSR :

    Moyennes par ligne :

        C1: (5+6+4)/3 = 5.00

        C2: (6+7+5)/3 = 6.00

    Moyennes par colonne :

        P1: (5+6)/2 = 5.50

        P2: (6+7)/2 = 6.50

        P3: (4+5)/2 = 4.50

    Moyenne globale :
    μ = (5+6+4+6+7+5)/6 = 33/6 = 5.50

    Calcul des résidus :
    Pour C1-P1 : 5 - 5.00 - 5.50 + 5.50 = 0.00
    Pour C1-P2 : 6 - 5.00 - 6.50 + 5.50 = 0.00
    Pour C1-P3 : 4 - 5.00 - 4.50 + 5.50 = 0.00

    Pour C2-P1 : 6 - 6.00 - 5.50 + 5.50 = 0.00
    Pour C2-P2 : 7 - 6.00 - 6.50 + 5.50 = 0.00
    Pour C2-P3 : 5 - 6.00 - 4.50 + 5.50 = 0.00

    MSR = 0 (parfait !)

Nous avons trouvé notre premier bicluster :

    Lignes : {C1, C2}

    Colonnes : {P1, P2, P3}

    Pattern : Les valeurs sont exactement +1 entre C1 et C2 pour chaque produit
    C1: [5, 6, 4]
    C2: [6, 7, 5] = C1 + 1

Étape 2 : Recherche d'un deuxième bicluster
Essai 3 : Examinons C3 et C4 avec P1, P2, P3
text

      P1  P2  P3
C3     9   8  10
C4     8   7   9

Calcul du MSR :

    Moyennes par ligne :

        C3: (9+8+10)/3 = 9.00

        C4: (8+7+9)/3 = 8.00

    Moyennes par colonne :

        P1: (9+8)/2 = 8.50

        P2: (8+7)/2 = 7.50

        P3: (10+9)/2 = 9.50

    Moyenne globale :
    μ = (9+8+10+8+7+9)/6 = 51/6 = 8.50

    Calcul des résidus :
    Pour C3-P1 : 9 - 9.00 - 8.50 + 8.50 = 0.00
    Pour C3-P2 : 8 - 9.00 - 7.50 + 8.50 = 0.00
    Pour C3-P3 : 10 - 9.00 - 9.50 + 8.50 = 0.00

    Pour C4-P1 : 8 - 8.00 - 8.50 + 8.50 = 0.00
    Pour C4-P2 : 7 - 8.00 - 7.50 + 8.50 = 0.00
    Pour C4-P3 : 9 - 8.00 - 9.50 + 8.50 = 0.00

    MSR = 0 (parfait !)

Deuxième bicluster :

    Lignes : {C3, C4}

    Colonnes : {P1, P2, P3}

    Pattern : C4 = C3 - 1 pour chaque produit

Étape 3 : Recherche d'un troisième bicluster (éventuellement chevauchant)
Essai 4 : Examinons C1, C2, C3, C4 avec P4, P5
text

      P4  P5
C1     7   8
C2     8   9
C3     7   6
C4     6   5

Calcul du MSR :

    Moyennes par ligne :

        C1: (7+8)/2 = 7.50

        C2: (8+9)/2 = 8.50

        C3: (7+6)/2 = 6.50

        C4: (6+5)/2 = 5.50

    Moyennes par colonne :

        P4: (7+8+7+6)/4 = 28/4 = 7.00

        P5: (8+9+6+5)/4 = 28/4 = 7.00

    Moyenne globale :
    μ = (7+8+8+9+7+6+6+5)/8 = 56/8 = 7.00

    Calcul des résidus :
    Pour C1-P4 : 7 - 7.50 - 7.00 + 7.00 = -0.50
    Pour C1-P5 : 8 - 7.50 - 7.00 + 7.00 = 0.50
    Pour C2-P4 : 8 - 8.50 - 7.00 + 7.00 = -0.50
    Pour C2-P5 : 9 - 8.50 - 7.00 + 7.00 = 0.50
    Pour C3-P4 : 7 - 6.50 - 7.00 + 7.00 = 0.50
    Pour C3-P5 : 6 - 6.50 - 7.00 + 7.00 = -0.50
    Pour C4-P4 : 6 - 5.50 - 7.00 + 7.00 = 0.50
    Pour C4-P5 : 5 - 5.50 - 7.00 + 7.00 = -0.50

    MSR = (0.25+0.25+0.25+0.25+0.25+0.25+0.25+0.25)/8 = 2.00/8 = 0.25

MSR = 0.25 (pas parfait, mais intéressant)

Si on regarde le pattern :

    C1 et C2 : valeurs élevées pour P4, P5 (7-8, 8-9)

    C3 et C4 : valeurs basses pour P4, P5 (7-6, 6-5)

Visualisation des Biclusters
text

      P1  P2  P3  P4  P5
C1    [5   6   4]  7   8
C2    [6   7   5]  8   9
C3    [9   8  10]  7   6
C4    [8   7   9]  6   5

Bicluster 1 : Encadré par [] - Lignes {C1,C2}, Colonnes {P1,P2,P3}
Bicluster 2 : Encadré par [] - Lignes {C3,C4}, Colonnes {P1,P2,P3}
Bicluster potentiel 3 : Lignes {C1,C2}, Colonnes {P4,P5} (valeurs élevées)
Bicluster potentiel 4 : Lignes {C3,C4}, Colonnes {P4,P5} (valeurs basses)
Interprétation des Résultats

Bicluster 1 (C1,C2 × P1,P2,P3) :

    Ces deux clients ont un pattern identique (+1 entre eux)

    Ils évaluent bien les produits P1, P2, P3

    Peut représenter un segment de marché : "clients modérés"

Bicluster 2 (C3,C4 × P1,P2,P3) :

    Pattern également cohérent (C4 = C3 - 1)

    Évaluations plus élevées pour P1,P2,P3 que le premier groupe

    Segment : "clients exigeants"

Pattern global :

    Les produits P1,P2,P3 montrent une structure claire entre clients

    Les produits P4,P5 séparent aussi les clients mais différemment

Applications Réelles
1. Analyse d'Expression Génique
text

Matrice : Gènes × Conditions expérimentales
Bicluster : Un groupe de gènes co-exprimés dans un sous-ensemble de conditions
→ Découverte de voies métaboliques actives seulement dans certaines conditions

2. Recommandation de Produits
text

Matrice : Clients × Produits (notes)
Bicluster : Un groupe de clients qui aiment un sous-ensemble de produits
→ Marketing ciblé : "Les clients qui aiment X aiment aussi Y"

3. Text Mining
text

Matrice : Documents × Mots (fréquence)
Bicluster : Un groupe de documents partageant un sous-ensemble de mots
→ Détection de thèmes communs dans un sous-ensemble de documents

