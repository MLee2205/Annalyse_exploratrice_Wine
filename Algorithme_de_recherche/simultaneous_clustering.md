## Simultaneous Clustering (Co-clustering / Bi-clustering simultané)

Le Simultaneous Clustering (ou Co-clustering) est une technique qui clusterise simultanément les lignes ET les colonnes
 d'une matrice de données. Contrairement au clustering traditionnel qui 
regroupe soit les lignes, soit les colonnes, le co-clustering trouve des
 blocs homogènes dans la matrice.

1. Concept Fondamental
Idée : Trouver une permutation des lignes et des colonnes telle que la matrice réorganisée montre des blocs homogènes.
Matrice originale :
Col1 Col2 Col3 Col4
[  5   8   2   1 ] Ligne1
[  7   9   3   2 ] Ligne2  
[  1   2   8   9 ] Ligne3
[  2   3   7   8 ] Ligne4
Après co-clustering :
text

Col2 Col1 Col4 Col3  (colonnes réordonnées)
[  8   5   1   2 ] Ligne1
[  9   7   2   3 ] Ligne2  → Bloc 1 (haut-gauche)
[  2   1   9   8 ] Ligne3
[  3   2   8   7 ] Ligne4  → Bloc 2 (bas-droite)
2. EXEMPLE À LA MAIN COMPLET
Matrice de données : Notes de films (1-5)


ActionRomanceComédieDocumentaireAlice5131Bob4222Claire1524David2435
Objectif : Trouver des groupes de personnes avec des préférences similaires pour des groupes de genres.
Étape 1 : Analyse visuelle initiale
text

       Action Romance Comédie Docu
Alice    5      1       3      1
Bob      4      2       2      2
Claire   1      5       2      4  
David    2      4       3      5
On remarque :

Alice et Bob : Hautes notes pour Action, basses pour Romance/Documentaire
Claire et David : Hautes notes pour Romance/Documentaire, basses pour Action
Étape 2 : Normalisation par ligne et colonne (simplifiée)
Pour comparer, on centre chaque ligne :
Centrage par ligne (soustraire la moyenne de la ligne) :

Alice : Moyenne = (5+1+3+1)/4 = 2.5
Action: 5-2.5 = 2.5
Romance: 1-2.5 = -1.5
Comédie: 3-2.5 = 0.5
Docu: 1-2.5 = -1.5
Bob : Moyenne = (4+2+2+2)/4 = 2.5
Action: 1.5, Romance: -0.5, Comédie: -0.5, Docu: -0.5
Claire : Moyenne = (1+5+2+4)/4 = 3.0
Action: -2.0, Romance: 2.0, Comédie: -1.0, Docu: 1.0
David : Moyenne = (2+4+3+5)/4 = 3.5
Action: -1.5, Romance: 0.5, Comédie: -0.5, Docu: 1.5
Matrice centrée :
text

       Action Romance Comédie Docu
Alice   2.5    -1.5     0.5   -1.5
Bob     1.5    -0.5    -0.5   -0.5  
Claire -2.0     2.0    -1.0    1.0
David  -1.5     0.5    -0.5    1.5
Étape 3 : Calcul de similarité entre lignes
Distance entre Alice et Bob :

Action: (2.5-1.5)² = 1.0
Romance: (-1.5+0.5)² = 1.0
Comédie: (0.5+0.5)² = 1.0
Docu: (-1.5+0.5)² = 1.0
Distance totale = √4 = 2.0
Distance entre Alice et Claire :

(2.5+2.0)² = 20.25
(-1.5-2.0)² = 12.25
(0.5+1.0)² = 2.25
(-1.5-1.0)² = 6.25
Distance = √41 ≈ 6.4
Matrice de distance :
text

      Alice Bob Claire David
Alice   0   2.0  6.4   5.7
Bob    2.0   0   5.8   4.9
Claire 6.4  5.8   0    2.2
David  5.7  4.9  2.2    0
Groupes de lignes :

Groupe 1 : Alice, Bob (distance faible)
Groupe 2 : Claire, David (distance faible)
Étape 4 : Calcul de similarité entre colonnes
Matrice transposée (centrée) :
text

       Alice  Bob  Claire David
Action  2.5   1.5   -2.0  -1.5
Romance -1.5 -0.5    2.0   0.5
Comédie  0.5 -0.5   -1.0  -0.5
Docu    -1.5 -0.5    1.0   1.5
Distance Action-Romance :

Alice: (2.5+1.5)² = 16.0
Bob: (1.5+0.5)² = 4.0
Claire: (-2.0-2.0)² = 16.0
David: (-1.5-0.5)² = 4.0
Distance = √40 ≈ 6.32
Distance Action-Comédie = √[(2.5-0.5)²+(1.5+0.5)²+(-2.0+1.0)²+(-1.5+0.5)²] = √(4+4+1+1)=√10≈3.16
Distance Action-Docu = √(16+4+9+9)=√38≈6.16
Distance Romance-Comédie = √(4+0+9+1)=√14≈3.74
Distance Romance-Docu = √(0+0+1+1)=√2≈1.41 ← Très proche!
Distance Comédie-Docu = √(4+0+4+4)=√12≈3.46
Groupes de colonnes :

Groupe A : Romance, Documentaire (distance faible)
Groupe B : Action
Groupe C : Comédie (intermédiaire)
Étape 5 : Réorganisation simultanée
Réorganisons :

Lignes : Groupe 1 (Alice, Bob), Groupe 2 (Claire, David)
Colonnes : Groupe A (Romance, Docu), Groupe B (Action), Groupe C (Comédie)
Nouvelle matrice :
text

       Romance Docu Action Comédie
Alice      1     1     5      3
Bob        2     2     4      2
Claire     5     4     1      2
David      4     5     2      3
Blocs identifiés :

Bloc 1 (Alice/Bob × Romance/Docu) : Valeurs basses (1-2)
Bloc 2 (Alice/Bob × Action) : Valeurs élevées (4-5)
Bloc 3 (Claire/David × Romance/Docu) : Valeurs élevées (4-5)
Bloc 4 (Claire/David × Action) : Valeurs basses (1-2)
Bloc 5 (Tous × Comédie) : Valeurs moyennes (2-3)
Étape 6 : Interprétation
Groupes de spectateurs :

Groupe 1 (Alice, Bob) : Aiment l'Action, n'aiment pas Romance/Docu
Groupe 2 (Claire, David) : Aiment Romance/Docu, n'aiment pas l'Action
Groupes de genres :

Groupe A (Romance, Documentaire) : Appréciés ensemble
Groupe B (Action) : Opposé à Romance/Docu
Groupe C (Comédie) : Neutre
Pattern découvert : Polarisation Action vs Romance/Documentaire

