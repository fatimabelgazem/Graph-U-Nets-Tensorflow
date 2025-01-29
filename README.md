# Graph Prediction Model

Ce projet implémente un modèle de prédiction de graphes utilisant des réseaux de neurones pour effectuer des tâches de classification sur des graphes de données. Le modèle est entraîné à partir de données spécifiques en utilisant une architecture de réseau de neurones graphiques (Graph Neural Network, GNN).
## Table des matières
  1. [Description](#description)
  2. [Prérequis](#prérequis)
  3. [Utilisation](#utilisation)
     - [Préparation des données](#préparation-des-données)
     - [Exécution de l'entraînement](#exécution-de-lentraînement)
     - [Évaluation du modèle](#évaluation-du-modèle)
  4. [Structure du projet](#structure-du-projet)
     - [Rôle de chaque fichier](#rôle-de-chaque-fichier)
  5. [Auteurs](#Auteurs)
  6. [Références](#Références)
  
  ---
  
  ## Description
  Ce projet implémente un modèle de prédiction basé sur des graphes en utilisant TensorFlow. Le modèle est utilisé pour des tâches telles que la classification de nœuds et la prédiction de propriétés de graphes. Le code utilise une architecture de réseau de neurones pour apprendre les représentations des graphes et les utiliser pour prédire des résultats.
  Le modèle s'entraîne sur un ensemble de données de graphes, qui peut être chargé à partir de fichiers externes. Les performances du modèle sont évaluées en utilisant des métriques telles que la précision et la perte.
  
---
## Prérequis
Avant d'exécuter ce projet, vous devez installer les dépendances suivantes :

  - **Python 3.7+**
  - **TensorFlow 2.x**
  - **NumPy**
  - **tqdm**
Vous pouvez installer les dépendances requises avec le fichier requirements.txt :
```bash
pip install -r requirements.txt
```
---
## Utilisation
### Préparation des données
Les données doivent être sous la forme de graphes. Le projet utilise un module appelé FileLoader pour charger les données dans le format attendu. Vous pouvez spécifier le chemin vers le dossier contenant les données avec l'argument -data.

### Exécution de l'entraînement
Une fois les données préparées, vous pouvez entraîner le modèle en utilisant le script main.py.

Voici un exemple de commande pour lancer l'entraînement :
```bash
python main.py -data PROTEINS -num_epochs 200 -batch 64 -lr 0.001 -fold 1
```
Cette commande lance l'entraînement sur le premier fold du jeu de données PROTEINS avec les paramètres suivants :

  - Nombre d'époques d'entraînement : 200
  - Taille des lots : 64
  - Taux d'apprentissage : 0.001
  - Fold : 1
    
Si vous ne spécifiez aucun argument, le modèle utilisera les valeurs par défaut définies dans main.py :
```bash
python main.py
```
Les paramètres par défaut sont :
  - num_epochs=200
  - batch_size=64
  - learning_rate=0.001
  - deg_as_tag=0
  - layer_num=3
  - hidden_dim=512
  - layer_dim=64
  - drop_network=0.3
  - drop_classifier=0.3
  - activation_network=ELU
  - activation_classifier=ELU
  - pool_rates_layers="0.9 0.8 0.7"
### Évaluation du modèle
Après l'entraînement, le modèle sera évalué sur le jeu de test, et les résultats seront enregistrés dans le fichier spécifié par l'argument -acc_file(re.txt). Le modèle utilise la précision et la perte pour évaluer ses performances.
Les résultats des performances seront affichés à la fin de chaque époque et sauvegardés dans un fichier pour un suivi ultérieur.

---

## Rôle de chaque fichier:
- **main.py** : Le script principal qui coordonne l'exécution du projet. Il charge les arguments de la ligne de commande, charge les données à l'aide de FileLoader, initialise le modèle de réseau de neurones et démarre le processus d'entraînement. Il gère également l'entraînement sur un ou plusieurs folds selon les spécifications.

- **trainer.py** : Contient la classe Trainer qui gère l'entraînement du modèle. Il contient des méthodes pour la gestion des époques d'entraînement, le calcul des pertes et des précisions, et l'application des gradients pour l'optimisation du modèle. Il est responsable de l'entraînement du modèle avec les données et d'évaluation de sa performance.

- **network.py** : Contient la définition du réseau de neurones utilisé pour la prédiction de graphes. La classe GNet définit la structure du modèle, notamment les couches du réseau, les fonctions d'activation, et la manière dont les graphes sont traités. C'est ici que le modèle de réseau de neurones est créé et configuré.

- **ops.py** : Ce fichier contient des fonctions utilitaires supplémentaires pour le projet. Il peut inclure des fonctions pour manipuler les graphes, effectuer des transformations de données, ou réaliser des calculs nécessaires à l'entraînement du modèle.

---

## Auteurs
Ce projet a été développé par :
 - Fatima Belgazem : fatimabelgazem1@gmail.com
 - Maryam Ajahoud  : maryamajahoud134@gmail.com

---

## Références
 - Graph U-Nets : https://arxiv.org/abs/1905.05178
