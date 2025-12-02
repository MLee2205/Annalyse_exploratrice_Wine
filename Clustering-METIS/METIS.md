## Exemple à la main de l'algorithme METIS
1. Problème : Partitionner un graphe en 2 parties

Graphe d'entrée (8 nœuds, 10 arêtes non dirigées, poids = 1) :
text

Nœuds: A, B, C, D, E, F, G, H
Arêtes: A-B, A-C, B-C, C-D, D-E, D-F, E-F, E-G, F-H, G-H
Objectif : Partitionner ce graphe en 2 partitions (k=2) de taille équilibrée, en minimisant les arêtes coupées.
2. Phase 1 : Coarsening (Agglomération)
Étape 1.1 : Matching initial (appariement lourd)

Nous allons fusionner les nœuds connectés par des arêtes "lourdes" (ici toutes les arêtes ont poids 1).

Heuristique : On cherche à fusionner des paires de nœuds qui partagent beaucoup d'arêtes communes.

Paires choisies (une possibilité) :

    (A, B) : Fusionner A et B → super-nœud AB

    (E, F) : Fusionner E et F → super-nœud EF

    (G, H) : Fusionner G et H → super-nœud GH

Graphe après fusion (5 nœuds) :
text

Super-nœuds: AB, C, D, EF, GH
Arêtes: 
AB-C (2 arêtes originales: A-C, B-C)
C-D (1 arête: C-D)
D-EF (1 arête: D-E, et D-F via D-E-F? Non, D-E direct)
EF-GH (2 arêtes: E-G, F-H)

Étape 1.2 : Calcul des nouveaux poids

Pour chaque arête entre super-nœuds, le poids est la somme des poids des arêtes originales :

    AB-C :

        A-C (poids 1) + B-C (poids 1) = 2

    C-D :

        C-D (poids 1) = 1

    D-EF :

        D-E (poids 1) = 1

        D-F (indirect) → Non, pas d'arête directe D-F

        Donc poids = 1

    EF-GH :

        E-G (poids 1) + F-H (poids 1) = 2

Graphe coarsené niveau 1 :
text

   AB(2)
    |
    C(1)--D(1)--EF(2)--GH

3. Phase 2 : Partitionnement du petit graphe

Maintenant, nous avons un graphe à 5 nœuds. Nous voulons le partitionner en 2 parties.

Algorithme simple (car graphe petit) :

    Calculer le degré total = 2+1+1+2 = 6

    Objectif : couper le moins d'arêtes possible

Essai 1 : Couper entre D et EF

    Partition 1 : {AB, C, D}

    Partition 2 : {EF, GH}

    Arêtes coupées : D-EF (poids 1)

Essai 2 : Couper entre C et D

    Partition 1 : {AB, C}

    Partition 2 : {D, EF, GH}

    Arêtes coupées : C-D (poids 1) + D-EF? Non, D et EF sont dans la même partition

    Total arêtes coupées : 1

Essai 3 : Couper entre AB et C

    Partition 1 : {AB}

    Partition 2 : {C, D, EF, GH}

    Arêtes coupées : AB-C (poids 2)

    Total : 2

Choix optimal : Couper entre D et EF (coût = 1)

Partitionnement initial :

    Partition 0 : {AB, C, D}

    Partition 1 : {EF, GH}

4. Phase 3 : Uncoarsening et Raffinement
Étape 3.1 : Projeter sur le graphe niveau 1

Partition actuelle :

    P0: AB, C, D

    P1: EF, GH

Étape 3.2 : Raffinement au niveau 1

Algorithme de Kernighan-Lin simplifié :

Pour chaque nœud, calculer le gain si on le déplace :

Gain = (arêtes coupées vers sa partition actuelle) - (arêtes coupées vers l'autre partition)

    Nœud AB (dans P0) :

        Arêtes : AB-C (poids 2, dans P0) → non coupée

        Si on déplace AB vers P1 : arête AB-C devient coupée (gain = 0 - 2 = -2)

    Nœud C (dans P0) :

        Arêtes : AB-C (2, P0) + C-D (1, P0)

        Si on déplace C vers P1 : AB-C coupée (2) + C-D coupée (1) = 3

        Gain = 0 - 3 = -3

    Nœud D (dans P0) :

        Arêtes : C-D (1, P0) + D-EF (1, coupée entre P0 et P1)

        Actuellement : 1 arête coupée (D-EF)

        Si on déplace D vers P1 : C-D coupée (1) + D-EF non coupée (0) = 1

        Gain = 1 - 1 = 0

    Nœud EF (dans P1) :

        Arêtes : D-EF (1, coupée) + EF-GH (2, P1)

        Actuellement : 1 arête coupée

        Si on déplace EF vers P0 : D-EF non coupée (0) + EF-GH coupée (2) = 2

        Gain = 1 - 2 = -1

    Nœud GH (dans P1) :

        Arêtes : EF-GH (2, P1)

        Si on déplace GH vers P0 : EF-GH coupée (2)

        Gain = 0 - 2 = -2

Aucun gain positif → On garde la partition telle quelle.
Étape 3.3 : Projeter sur le graphe original

Maintenant, décomposons les super-nœuds :

    AB (P0) → A(P0), B(P0)

    C (P0) → C(P0)

    D (P0) → D(P0)

    EF (P1) → E(P1), F(P1)

    GH (P1) → G(P1), H(P1)

Partition projetée :

    Partition 0 : {A, B, C, D}

    Partition 1 : {E, F, G, H}

Arêtes coupées :

    C-D (non, les deux dans P0)

    D-E (OUI, D∈P0, E∈P1) → poids 1

    D-F? Non, pas d'arête directe

    Total arêtes coupées = 1

Étape 3.4 : Raffinement au niveau original

Calculons les gains pour chaque nœud individuel :

Formule : Pour un nœud X dans la partition P :
text

Gain(X) = Σ poids(arêtes de X vers l'autre partition) - Σ poids(arêtes de X vers sa propre partition)

    Nœud A (P0) :

        Arêtes : A-B (1, P0), A-C (1, P0)

        Gain si on déplace A vers P1 : (0) - (1+1) = -2

    Nœud B (P0) :

        Arêtes : A-B (1, P0), B-C (1, P0)

        Gain = 0 - 2 = -2

    Nœud C (P0) :

        Arêtes : A-C (1, P0), B-C (1, P0), C-D (1, P0)

        Gain = 0 - 3 = -3

    Nœud D (P0) :

        Arêtes : C-D (1, P0), D-E (1, coupée)

        Actuellement : 1 arête coupée

        Si on déplace D vers P1 :

            D-E deviendrait non coupée (0)

            C-D deviendrait coupée (1)

            Gain = 1 - 1 = 0

    Nœud E (P1) :

        Arêtes : D-E (1, coupée), E-F (1, P1), E-G (1, P1)

        Actuellement : 1 arête coupée

        Si on déplace E vers P0 :

            D-E non coupée (0)

            E-F coupée (1), E-G coupée (1)

            Gain = 1 - 2 = -1

    Nœud F (P1) :

        Arêtes : E-F (1, P1), F-H (1, P1)

        Gain = 0 - 2 = -2

    Nœud G (P1) :

        Arêtes : E-G (1, P1), G-H (1, P1)

        Gain = 0 - 2 = -2

    Nœud H (P1) :

        Arêtes : F-H (1, P1), G-H (1, P1)

        Gain = 0 - 2 = -2

Tous les gains sont ≤ 0 → Pas d'amélioration possible.
5. Résultat Final

Partitionnement final :

    Partition 0 : {A, B, C, D} (4 nœuds)

    Partition 1 : {E, F, G, H} (4 nœuds)

Arêtes coupées : Seulement D-E (poids 1)

Équilibre : Parfait (4-4)

Qualité : Excellente (1 seule arête coupée sur 10 totales)
6. Vérification Visuelle
text

   A(0)        E(1)
  / \          / \
 B(0)--C(0)--D(0) |--F(1)
                \ |   |
                 \|   |
                 G(1)-H(1)

Arêtes coupées marquées par --- entre partitions :

    Seulement D---E est coupée

7. Analyse de la Solution

Pour ce graphe, la solution trouvée est optimale :

    Coupe minimale : On doit couper au moins 1 arête car le graphe est connexe

    Équilibre parfait : 4-4

    Structure naturelle : Le graphe a un "goulot d'étranglement" entre D et E

8. Comparaison avec d'autres partitions possibles

Alternative 1 : Couper entre C et D

    P0: {A, B, C} (3 nœuds)

    P1: {D, E, F, G, H} (5 nœuds)

    Arêtes coupées : C-D (1) + D-E? Non, D et E dans P1

    Total : 1 arête, mais déséquilibre 3-5

Alternative 2 : Couper entre A-B et B-C

    P0: {A} (1 nœud)

    P1: {B, C, D, E, F, G, H} (7 nœuds)

    Arêtes coupées : A-B (1) + A-C (1) = 2

    Pire : plus d'arêtes coupées et très déséquilibré

Notre solution METIS est donc la meilleure en combinant équilibre et faible coupe.
9. Points Clés à Retenir

    Approche multi-niveaux : Réduction progressive + partitionnement + raffinement

    Heuristique d'appariement : Fusionne les nœuds fortement connectés

    Raffinement local : Algorithme de type Kernighan-Lin pour améliorer la partition

    Équilibrage : METIS contrôle naturellement l'équilibre de taille

    Efficacité : Même pour ce petit exemple, on voit la logique qui s'adapte aux grands graphes

Complexité : Pour ce graphe de 8 nœuds, METIS a fait :

    1 niveau de coarsening (8 → 5 nœuds)

    Partitionnement d'un graphe à 5 nœuds (trivial)

    2 niveaux de raffinement

Pour un graphe avec 1 million de nœuds, METIS ferait ~log(n) niveaux (~20 niveaux), chaque niveau étant 2x plus petit, d'où son efficacité O(n)

