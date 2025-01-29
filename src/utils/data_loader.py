import random
import tensorflow as tf
import networkx as nx
import numpy as np
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm
from functools import partial

# Classe représentant un ensemble de données de graphes
class GData:
    def __init__(self, num_class, feat_dim, g_list):
        """
        Initialise les données de graphes.

        :param num_class: Nombre de classes (labels uniques).
        :param feat_dim: Dimension des caractéristiques des nœuds.
        :param g_list: Liste des graphes.
        """
        self.num_class = num_class
        self.feat_dim = feat_dim
        self.g_list = g_list
        self.sep_data()  # Sépare les données en fonction des folds pour la validation croisée.

    def sep_data(self, seed=0):
        """
        Sépare les données en folds pour la validation croisée.

        :param seed: Permet de garantir que la séparation soit reproductible.
        """
        skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
        labels = [g.label for g in self.g_list]
        self.idx_list = list(skf.split(np.zeros(len(labels)), labels))

    def use_fold_data(self, fold_idx):
        """
        Sélectionne les données d'entraînement et de test pour un fold spécifique.

        :param fold_idx: Index du fold à utiliser pour la validation croisée.
        """
        self.fold_idx = fold_idx + 1
        train_idx, test_idx = self.idx_list[fold_idx]
        self.train_gs = [self.g_list[i] for i in train_idx]  # Graphes d'entraînement
        self.test_gs = [self.g_list[i] for i in test_idx]  # Graphes de test


# Classe pour le chargement et le traitement des fichiers de données de graphes
class FileLoader:
    def __init__(self, args):
        """
        Initialise le chargeur de fichiers.

        :param args: Arguments nécessaires pour le chargement des données.
        """
        self.args = args

    def line_genor(self, lines):
        """
        Générateur pour parcourir les lignes d'un fichier.

        :param lines: Liste de lignes du fichier.
        """
        for line in lines:
            yield line

    def gen_graph(self, f, i, label_dict, feat_dict, deg_as_tag):
        """
        Génère un graphe à partir des données du fichier.

        :param f: Générateur de lignes.
        :param i: Index du graphe.
        :param label_dict: Dictionnaire pour associer les labels aux graphes.
        :param feat_dict: Dictionnaire pour associer les caractéristiques aux nœuds.
        :param deg_as_tag: Utilisation du degré comme étiquette si True.
        :return: Graphe de type NetworkX avec les attributs nécessaires.
        """
        row = next(f).strip().split()
        n, label = [int(w) for w in row]
        if label not in label_dict:
            label_dict[label] = len(label_dict)

        g = nx.Graph()  # Création d'un graphe vide
        g.add_nodes_from(range(n))  # Ajout des nœuds
        node_tags = []  # Liste pour stocker les tags des nœuds

        for j in range(n):
            row = next(f).strip().split()
            tmp = int(row[1]) + 2
            row = [int(w) for w in row[:tmp]]
            if row[0] not in feat_dict:
                feat_dict[row[0]] = len(feat_dict)
            for k in range(2, len(row)):
                if j != row[k]:
                    g.add_edge(j, row[k])  # Ajout des arêtes entre nœuds
            if len(row) > 2:
                node_tags.append(feat_dict[row[0]])

        g.label = label  # Associe un label au graphe
        g.remove_nodes_from(list(nx.isolates(g)))  # Retire les nœuds isolés
        g.node_tags = list(dict(g.degree).values()) if deg_as_tag else node_tags
        return g

    def process_g(self, label_dict, tag2index, tagset, g):
        """
        Prépare un graphe pour l'entraînement en convertissant les tags et en ajoutant la matrice d'adjacence.

        :param label_dict: Dictionnaire des labels.
        :param tag2index: Dictionnaire de conversion des tags en indices.
        :param tagset: Ensemble des tags possibles.
        :param g: Graphe à traiter.
        :return: Graphe préparé pour l'entraînement.
        """
        g.label = label_dict[g.label]
        g.feas = tf.convert_to_tensor([tag2index[tag] for tag in g.node_tags], dtype=tf.int32)
        g.feas = tf.one_hot(g.feas, depth=len(tagset))  # Transformation en vecteurs one-hot
        A = tf.convert_to_tensor(nx.to_numpy_array(g), dtype=tf.float32)  # Conversion de la matrice d'adjacence en tensor

        g.A = A + tf.eye(g.number_of_nodes())  # Ajout de boucles auto-référentielles aux nœuds
        return g

    def load_data(self):
        """
        Charge les données depuis un fichier et les prépare.

        :return: Instance de GData contenant les graphes et les métadonnées.
        """
        args = self.args
        print('Loading data...')
        g_list = []
        label_dict = {}
        feat_dict = {}

        # Ouverture du fichier de données
        with open(f'data/{args.data}/{args.data}.txt', 'r') as f:
            lines = f.readlines()
        f = self.line_genor(lines)
        n_g = int(next(f).strip())  # Nombre de graphes

        for i in tqdm(range(n_g), desc="Create graph", unit='graphs'):
            g = self.gen_graph(f, i, label_dict, feat_dict, args.deg_as_tag)
            g_list.append(g)

        tagset = set()
        for g in g_list:
            tagset = tagset.union(set(g.node_tags))  # Collecte tous les tags
        tagset = list(tagset)
        tag2index = {tag: idx for idx, tag in enumerate(tagset)}  # Conversion des tags en indices

        # Application du traitement sur tous les graphes
        f_n = partial(self.process_g, label_dict, tag2index, tagset)
        new_g_list = []
        for g in tqdm(g_list, desc="Process graph", unit='graphs'):
            new_g_list.append(f_n(g))

        num_class = len(label_dict)
        feat_dim = len(tagset)

        print(f'# Classes: {num_class}, # Maximum node tag: {feat_dim}')
        return GData(num_class, feat_dim, new_g_list)
