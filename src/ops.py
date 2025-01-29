import tensorflow as tf
import numpy as np

# Classe GraphUnet qui hérite de tf.keras.Model, représentant le modèle Graph U-Net
class GraphUnet(tf.keras.Model):

    def __init__(self, ks, in_dim, out_dim, dim, act, drop_p):
        super(GraphUnet, self).__init__()
        # Initialisation des paramètres
        self.ks = ks  # Taille des filtres pour les couches de pooling
        self.bottom_gcn = GCN(dim, dim, act, drop_p)  # GCN au bas du réseau (pour la partie "bottleneck")
        self.down_gcns = []  # Liste pour les couches GCN lors de la phase descendante
        self.up_gcns = []  # Liste pour les couches GCN lors de la phase ascendante
        self.pools = []  # Liste des opérations de pooling
        self.unpools = []  # Liste des opérations de unpooling
        self.l_n = len(ks)  # Nombre de couches (égal à la longueur de ks)

        # Initialisation des couches pour chaque niveau de l'U-Net
        for _ in range(self.l_n):
            self.down_gcns.append(GCN(dim, dim, act, drop_p))  # Couches GCN descendantes
            self.up_gcns.append(GCN(dim, dim, act, drop_p))  # Couches GCN ascendantes
            self.pools.append(Pool(ks[_], dim, drop_p))  # Couches de pooling
            self.unpools.append(Unpool())  # Couches de unpooling

    # Fonction principale de propagation avant (forward pass)
    def call(self, g, h):
        adj_ms = []  # Liste pour stocker les graphes
        indices_list = []  # Liste pour les indices des pools
        down_outs = []  # Liste pour stocker les sorties de chaque étape descendante
        hs = []  # Liste pour stocker les sorties à chaque étape du réseau
        org_h = h  # Sauvegarde de l’entrée initiale pour la connexion résiduelle à la fin

        # Phase descendante (downsampling)
        for i in range(self.l_n):
            h = self.down_gcns[i](g, h)  # Application de la couche GCN
            adj_ms.append(g)  # Sauvegarde du graphe
            down_outs.append(h)  # Sauvegarde de la sortie de la couche
            g, h, idx = self.pools[i](g, h)  # Pooling
            indices_list.append(idx)  # Sauvegarde des indices pour le unpooling

        # Partie centrale (bottleneck) avec une seule couche GCN
        h = self.bottom_gcn(g, h)

        # Phase ascendante (upsampling)
        for i in range(self.l_n):
            up_idx = self.l_n - i - 1  # Index pour remonter dans le modèle
            g, idx = adj_ms[up_idx], indices_list[up_idx]  # Récupère le graphe et les indices de pooling
            g, h = self.unpools[i](g, h, down_outs[up_idx], idx)  # Unpooling
            h = self.up_gcns[i](g, h)  # Application de la couche GCN ascendante
            h = h + down_outs[up_idx]  # Ajout de la connexion résiduelle
            hs.append(h)  # Sauvegarde de la sortie

        # Connexion résiduelle avec l'entrée initiale
        h = h + org_h
        hs.append(h)  # Sauvegarde de la sortie finale
        return hs


# Classe GCN (Graph Convolutional Network), une couche de convolution pour le graphe
class GCN(tf.keras.layers.Layer):

    def __init__(self, in_dim, out_dim, act, drop_p):
        super(GCN, self).__init__()
        self.proj = tf.keras.layers.Dense(out_dim)  # Couche de projection (Dense)
        self.act = act  # Fonction d'activation
        self.drop = tf.keras.layers.Dropout(rate=drop_p) if drop_p > 0.0 else tf.keras.layers.Layer()  # Dropout optionnel

    # Fonction de propagation avant (forward pass) pour le GCN
    def call(self, g, h):
        print(f"Executing call, eager mode: {tf.executing_eagerly()}")
        print(f"Input h shape: {h.shape}")

        h = self.drop(h)  # Applique le dropout

        # Vérification de la dimension de h
        if len(h.shape) == 3:  # Si h a une dimension batch en trop
            h = tf.squeeze(h, axis=0)  # Supprime la dimension supplémentaire

        print(f"Processed h shape: {h.shape}")

        # Vérification de la forme de h avant multiplication matricielle
        if h.shape[0] == 1:  # Cas problématique
            h = tf.reshape(h, (32, -1))  # Reformater en (32, X)

        print(f"Reshaped h for MatMul: {h.shape}")

        # Calcul du produit matriciel entre g et h
        h = tf.matmul(g, h)
        h = self.proj(h)  # Applique la projection
        h = self.act(h)  # Applique la fonction d'activation
        
        return h


# Classe Pool pour effectuer le pooling sur le graphe
class Pool(tf.keras.layers.Layer):

    def __init__(self, k, in_dim, drop_p):
        super(Pool, self).__init__()
        self.k = k  # Taille du top-k pour le pooling
        self.sigmoid = tf.keras.activations.sigmoid  # Fonction d'activation sigmoid
        self.proj = tf.keras.layers.Dense(1)  # Couche de projection pour calculer les poids
        self.drop = tf.keras.layers.Dropout(rate=drop_p) if drop_p > 0 else tf.keras.layers.Layer()  # Dropout optionnel

    def call(self, g, h):
        Z = self.drop(h)  # Applique le dropout sur h
        weights = tf.squeeze(self.proj(Z), axis=-1)  # Applique la projection
        scores = self.sigmoid(weights)  # Applique la fonction sigmoid pour obtenir les scores
        return top_k_graph(scores, g, h, self.k)  # Applique le pooling basé sur les scores


# Classe Unpool pour effectuer l'unpooling sur le graphe
class Unpool(tf.keras.layers.Layer):

    def call(self, g, h, pre_h, idx):
        new_h = tf.zeros([tf.shape(g)[0], tf.shape(h)[1]])  # Initialisation d'un vecteur vide
        new_h = tf.tensor_scatter_nd_update(new_h, tf.expand_dims(idx, axis=-1), h)  # Applique l'unpooling
        return g, new_h


# Fonction pour effectuer un pooling top-k sur le graphe
def top_k_graph(scores, g, h, k):
    num_nodes = tf.shape(g)[0]  # Nombre de nœuds dans le graphe
    k_val = tf.maximum(2, tf.cast(k * tf.cast(num_nodes, tf.float32), tf.int32))  # Détermine le nombre de voisins pour le top-k
    values, idx = tf.math.top_k(scores, k=k_val)  # Applique le top-k sur les scores
    new_h = tf.gather(h, idx)  # Récupère les éléments correspondant aux indices du top-k
    values = tf.expand_dims(values, axis=-1)
    new_h = new_h * values  # Applique les valeurs du top-k aux nouvelles caractéristiques

    # Normalisation du graphe
    un_g = tf.cast(tf.cast(g, tf.bool), tf.float32)
    un_g = tf.cast(tf.matmul(un_g, un_g) > 0, tf.float32)
    un_g = tf.gather(un_g, idx, axis=0)
    un_g = tf.gather(un_g, idx, axis=1)
    g = norm_g(un_g)  # Normalisation des liens du graphe
    return g, new_h, idx


# Fonction de normalisation du graphe
def norm_g(g):
    degrees = tf.reduce_sum(g, axis=1, keepdims=True)  # Somme des degrés des nœuds
    g = g / degrees  # Normalisation des liens
    return g


# Initialisation des poids (Glorot Uniform)
class Initializer(object):

    @classmethod
    def glorot_uniform(cls, shape):
        fan_in, fan_out = shape[-2], shape[-1]  # Nombre d'entrées et de sorties pour le calcul de l'initialisation
        limit = np.sqrt(6.0 / (fan_in + fan_out))  # Calcul de la limite de Glorot
        return tf.random.uniform(shape, minval=-limit, maxval=limit)  # Initialisation uniforme

    @classmethod
    def param_init(cls, layer):
        if isinstance(layer, tf.keras.layers.Dense):
            layer.kernel_initializer = tf.keras.initializers.GlorotUniform()  # Initialisation des poids
            layer.bias_initializer = tf.keras.initializers.Zeros()  # Initialisation des biais à zéro

    @classmethod
    def weights_init(cls, model):
        for layer in model.layers:
            cls.param_init(layer)  # Applique l'initialisation à chaque couche du modèle
