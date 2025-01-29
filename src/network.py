import tensorflow as tf
from tensorflow.keras import layers, models
from ops import GCN, GraphUnet, Initializer, norm_g

# Définition du modèle GNet
class GNet(tf.keras.Model):
    def __init__(self, in_dim, n_classes, args):
        # Initialisation du modèle
        super(GNet, self).__init__()
        # Récupération des fonctions d'activation depuis tf.nn en fonction des paramètres
        self.n_act = getattr(tf.nn, args.act_n)  # Activation pour le GCN (Graph Convolution Network)
        self.c_act = getattr(tf.nn, args.act_c)  # Activation pour la classification
        
        # Initialisation des couches du GCN avec les paramètres d'entrée
        self.s_gcn = GCN(in_dim, args.l_dim, self.n_act, args.drop_n)
        
        # Initialisation du GraphUnet avec les paramètres donnés
        self.g_unet = GraphUnet(
            args.ks, args.l_dim, args.l_dim, args.l_dim, self.n_act, args.drop_n)
        
        # Couche de sortie 1 : Dense pour transformer les représentations en taille h_dim
        self.out_l_1 = tf.keras.layers.Dense(args.h_dim)
        # Couche de sortie 2 : Dense pour classer en n_classes (nombre de classes)
        self.out_l_2 = tf.keras.layers.Dense(n_classes)
        
        # Couche Dropout pour réduire le surapprentissage
        self.out_drop = tf.keras.layers.Dropout(args.drop_c)
        
        # Initialisation des poids des couches du modèle
        Initializer.weights_init(self)

    # Fonction d'appel du modèle
    def call(self, gs, hs, labels):
        # Embedding des graphes et des features (les nœuds)
        hs = self.embed(gs, hs)
        # Classification des embeddings obtenus
        logits = self.classify(hs)
        # Calcul des métriques comme la perte et la précision
        return self.metric(logits, labels)

    # Fonction pour faire l'embedding des graphes (par les différentes couches)
    def embed(self, gs, hs):
        o_hs = []
        # Applique l'embed pour chaque couple de graphe et feature
        for g, h in zip(gs, hs):
            h = self.embed_one(g, h)
            o_hs.append(h)
        # Empile les résultats pour former le batch final
        hs = tf.stack(o_hs, axis=0)
        return hs

    # Embedding pour un seul graphe et ses features
    def embed_one(self, g, h):
        # Normalisation du graphe (on normalise la matrice d'adjacence)
        g = norm_g(g)
        # Application de la couche GCN sur le graphe et ses features
        h = self.s_gcn(g, h)
        # Passage dans le GraphUnet pour extraire les représentations
        hs = self.g_unet(g, h)
        # Effectuer le "readout" pour agréger les informations sur les nœuds
        h = self.readout(hs)
        return h

    # Agrégation des résultats du GraphUnet
    def readout(self, hs):
        # Calcul des différentes statistiques sur les sorties de GraphUnet
        h_max = [tf.reduce_max(h, axis=0) for h in hs]  # Max sur chaque feature
        h_sum = [tf.reduce_sum(h, axis=0) for h in hs]  # Somme sur chaque feature
        h_mean = [tf.reduce_mean(h, axis=0) for h in hs]  # Moyenne sur chaque feature
        # Concaténation des résultats pour former un seul vecteur de sortie
        h = tf.concat(h_max + h_sum + h_mean, axis=0)
        return h

    # Fonction de classification
    def classify(self, h):
        # Application de Dropout pour réduire le surapprentissage
        h = self.out_drop(h)
        # Passer à travers la première couche Dense
        h = self.out_l_1(h)
        # Activation de la classification
        h = self.c_act(h)
        # Nouveau passage dans Dropout
        h = self.out_drop(h)
        # Dernière couche Dense pour produire les logits finaux
        h = self.out_l_2(h)
        return tf.nn.log_softmax(h, axis=1)  # Retourne les logits après log softmax

    # Calcul de la perte (cross-entropy) et de la précision (accuracy)
    def metric(self, logits, labels):
        # Calcul de la perte avec la fonction softmax cross-entropy
        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels))
        # Prédictions basées sur les logits
        preds = tf.argmax(logits, axis=1)
        # Calcul de la précision
        acc = tf.reduce_mean(tf.cast(tf.equal(preds, labels), tf.float32))
        return loss, acc
