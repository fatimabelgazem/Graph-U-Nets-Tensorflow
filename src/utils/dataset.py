import random
import tensorflow as tf


# Classe représentant les données de graphes pour l'entraînement
class GraphData:
    def __init__(self, data, feat_dim):
        """
        Initialise l'objet GraphData.

        :param data: Liste des graphes. Chaque élément doit avoir une matrice d'adjacence, des caractéristiques, et un label.
        :param feat_dim: Dimension des caractéristiques des nœuds.
        """
        self.data = data
        self.feat_dim = feat_dim  # Dimension des caractéristiques des nœuds
        self.idx = list(range(len(data)))  # Liste des indices des graphes
        self.pos = 0  # Position actuelle dans l'itération
        self.batch = None  # Taille des batchs (sera définie par loader())
        self.shuffle = False  # Si True, les données seront mélangées avant chaque itération

    def __reset__(self):
        """
        Réinitialise la position et mélange les indices si nécessaire.
        """
        self.pos = 0
        if self.shuffle:
            random.shuffle(self.idx)

    def __len__(self):
        """
        Retourne le nombre de batchs dans les données.
        """
        return len(self.data) // self.batch + 1

    def __getitem__(self, idx):
        """
        Récupère un graphe à partir d'un index donné.

        :param idx: Index du graphe à récupérer.
        :return: Matrice d'adjacence (A), caractéristiques (features), et label.
        """
        g = self.data[idx]
        return g.A, tf.convert_to_tensor(g.feas, dtype=tf.float32), g.label

    def __iter__(self):
        """
        Rend l'objet itérable.
        """
        return self

    def __next__(self):
        """
        Récupère le prochain batch de graphes.

        :return: Nombre de graphes, matrices d'adjacence, caractéristiques et labels pour le batch actuel.
        """
        if self.pos >= len(self.data):
            self.__reset__()
            raise StopIteration  # Signale la fin de l'itération

        cur_idx = self.idx[self.pos: self.pos + self.batch]  # Récupère les indices pour le batch actuel
        data = [self.__getitem__(idx) for idx in cur_idx]  # Chargement des données pour ce batch
        self.pos += len(cur_idx)

        # Décompresse les données en trois listes : matrices d'adjacence, caractéristiques, et labels
        gs, hs, labels = map(list, zip(*data))

        return len(gs), gs, hs, tf.convert_to_tensor(labels, dtype=tf.int64)

    def loader(self, batch, shuffle, *args):
        """
        Configure le générateur pour charger les données en batchs.

        :param batch: Taille des batchs.
        :param shuffle: Si True, les données seront mélangées.
        :return: L'objet GraphData configuré.
        """
        self.batch = batch
        self.shuffle = shuffle
        if shuffle:
            random.shuffle(self.idx)
        return self
