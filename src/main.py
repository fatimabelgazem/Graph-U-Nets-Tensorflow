import argparse
import random
import time
import tensorflow as tf
import numpy as np
from network import GNet  # Importation du réseau de neurones défini dans le fichier network.py
from trainer import Trainer  # Importation de la classe gérant l'entraînement défini dans le fichier trainer.py
from utils.data_loader import FileLoader  # Importation du chargeur de données défini dans le fichier data_loader.py 

# Fonction pour définir et récupérer les arguments de la ligne de commande
def get_args():
    parser = argparse.ArgumentParser(description='Args for graph prediction')  # Initialisation du parser
    # Définition des arguments acceptés par le script
    parser.add_argument('-seed', type=int, default=1, help='Seed aléatoire pour la reproductibilité')
    parser.add_argument('-data', default='PROTEINS', help='Nom du dossier contenant les données')
    parser.add_argument('-fold', type=int, default=1, help='Numéro de la division des données (1 à 10)')
    parser.add_argument('-num_epochs', type=int, default=200, help='Nombre d\'époques d\'entraînement')
    parser.add_argument('-batch', type=int, default=64, help='Taille des lots')
    parser.add_argument('-lr', type=float, default=0.001, help='Taux d\'apprentissage')
    parser.add_argument('-deg_as_tag', type=int, default=0, help='Utiliser le degré du nœud comme tag (1 pour oui)')
    parser.add_argument('-l_num', type=int, default=3, help='Nombre de couches du réseau')
    parser.add_argument('-h_dim', type=int, default=512, help='Dimension des couches cachées')
    parser.add_argument('-l_dim', type=int, default=64, help='Dimension de la couche de sortie')
    parser.add_argument('-drop_n', type=float, default=0.3, help='Taux de dropout pour le réseau')
    parser.add_argument('-drop_c', type=float, default=0.3, help='Taux de dropout pour la sortie')
    parser.add_argument('-act_n', type=str, default='elu', help='Fonction d\'activation pour le réseau')
    parser.add_argument('-act_c', type=str, default='elu', help='Fonction d\'activation pour la sortie')
    parser.add_argument('-ks', nargs='+', type=float, default=[0.9, 0.8, 0.7], help='Liste de k pour le pooling')
    parser.add_argument('-acc_file', type=str, default='re', help='Nom du fichier pour sauvegarder les résultats')
    args, _ = parser.parse_known_args()  # Analyse les arguments
    return args

# Fonction pour fixer la seed aléatoire pour assurer la reproductibilité
def set_random(seed):
    random.seed(seed)  # Seed pour les opérations Python
    np.random.seed(seed)  # Seed pour NumPy
    tf.random.set_seed(seed)  # Seed pour TensorFlow

# Fonction principale pour l'exécution d'une session d'entraînement sur un fold spécifique
def app_run(args, G_data, fold_idx):
    G_data.use_fold_data(fold_idx)  # Charge les données du fold spécifique
    net = GNet(G_data.feat_dim, G_data.num_class, args)  # Initialise le modèle de réseau de neurones
    trainer = Trainer(args, net, G_data)  # Initialise le gestionnaire d'entraînement
    trainer.train()  # Lance l'entraînement

# Fonction principale de l'application
def main():
    args = get_args()  # Récupère les arguments de la ligne de commande
    print(args)  # Affiche les arguments pour vérification
    set_random(args.seed)  # Fixe la seed pour la reproductibilité

    start = time.time()  # Démarre le chronomètre pour mesurer le temps de chargement des données
    G_data = FileLoader(args).load_data()  # Charge les données depuis le dossier spécifié
    print('Load data using ------>', time.time() - start)  # Affiche le temps de chargement des données

    # Vérifie si l'entraînement doit être effectué sur tous les folds ou un fold spécifique
    if args.fold == 0:
        # Entraînement sur tous les folds (10 folds par convention)
        for fold_idx in range(10):
            print('Start training ------> fold', fold_idx + 1)
            app_run(args, G_data, fold_idx)  # Lance l'entraînement pour chaque fold
    else:
        # Entraînement sur un fold spécifique
        print('Start training ------> fold', args.fold)
        app_run(args, G_data, args.fold - 1)  # Lance l'entraînement pour un seul fold

# Point d'entrée du script
if __name__ == "__main__":
    main()  # Exécution de la fonction principale
