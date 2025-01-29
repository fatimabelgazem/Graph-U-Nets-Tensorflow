---

# Description  du projet

  Ce projet porte sur le traitement et l'utilisation de graphes pour la classification dans le cadre de modèles d'apprentissage profond. Les fichiers fournis, data_loader.py et dataset.py, contiennent des outils pour charger et traiter des graphes à partir de données brutes, les préparer pour l'entraînement, et gérer des batchs de données dans des modèles de réseaux de neurones.

---

# Description des fichiers
## 1. data_loader.py
Le fichier data_loader.py contient la classe FileLoader, qui est responsable du chargement des données depuis un fichier et de la création des graphes à partir de ces données. Il inclut également la classe GData qui organise les graphes en ensembles de données pour l'entraînement et la validation.

### Fonctionnalités principales :

Chargement des graphes à partir d'un fichier : Le fichier génère des graphes en utilisant des matrices d'adjacence et des caractéristiques pour chaque nœud.
Préparation des graphes : Les graphes sont traités pour être utilisés dans un modèle de classification (conversion en tensors, ajout de connexions pour les boucles, etc.).
Séparation des données en folds : La classe GData permet de séparer les données en plusieurs folds pour effectuer une validation croisée.
### Classe principale :

FileLoader : Charge les graphes et prépare les données pour l'entraînement.
GData : Contient les graphes et gère la séparation en folds pour la validation croisée.

## 2. dataset.py

Le fichier dataset.py contient la classe GraphData, qui permet de gérer les graphes et leurs données associées pour l'entraînement dans un modèle d'apprentissage supervisé. Cette classe est responsable de l'itération sur les graphes et de la création de batchs pour l'entraînement du modèle.

### Fonctionnalités principales :

Gestion des graphes : La classe GraphData prend une liste de graphes, chacun avec sa matrice d'adjacence, ses caractéristiques de nœuds, et son label.
Gestion des batchs : La classe permet de créer des batchs de données pour l'entraînement en les parcourant de manière itérative.
### Classe principale :

GraphData : Gère les graphes, leurs caractéristiques, et leurs labels pour l'entraînement.
Description du Dataset
Les données utilisées dans ce projet sont des graphes représentant des entités connectées. Chaque graphe est constitué de nœuds (entités) et d'arêtes (relations entre ces entités). Les nœuds possèdent des caractéristiques qui peuvent être utilisées pour la classification des graphes. Chaque graphe possède également un label qui peut être utilisé pour l'entraînement d'un modèle de classification.

# Exemple de dataset :

Le dataset utilisé dans ce projet est un dataset typique de classification de graphes, comme le dataset PROTEINS, qui contient des graphes représentant des structures de protéines. Chaque graphe représente une protéine où les nœuds correspondent aux résidus d'acides aminés et les arêtes indiquent les interactions entre ces résidus. Chaque protéine est classée dans une des classes définies par le dataset.

Les graphes sont représentés par :

Matrice d'adjacence (A) : Représente les connexions entre les résidus d'acides aminés dans la protéine.
Caractéristiques des nœuds : Chaque nœud possède des attributs qui décrivent certaines propriétés des résidus d'acides aminés, comme leur structure ou leur fonction.
Label : Chaque graphe est associé à une classe qui peut représenter une catégorie fonctionnelle ou structurelle de la protéine.

# Format des données TXT

1ère ligne : N, qui représente le nombre de graphes. Les N blocs suivants décrivent chacun un graphe.

Pour chaque bloc de texte :

 - Une ligne contient n l, où :
     - n est le nombre de nœuds dans le graphe actuel,
     - l est le label (étiquette) du graphe.
 - Les n lignes suivantes décrivent les nœuds du graphe :
    - La ième ligne (indexée à partir de 0) commence par t m, où :
       - t est le tag du nœud actuel,
       - m est le nombre de voisins du nœud actuel.
       - Les m nombres suivants indiquent les indices des nœuds voisins (commençant à 0).
       - Les d nombres suivants (s'il y en a) représentent les caractéristiques continues du nœud (attributs).
