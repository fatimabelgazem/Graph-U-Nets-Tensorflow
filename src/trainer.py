import tensorflow as tf
from tqdm import tqdm
from utils.dataset import GraphData

# Classe principale pour l'entraînement du modèle
class Trainer:
    def __init__(self, args, net, G_data):
        """
        Initialisation de la classe Trainer.

        Arguments:
            args : Arguments contenant les hyperparamètres et les configurations.
            net : Le modèle de réseau de neurones à entraîner (le modèle GraphNet).
            G_data : Données de graphes utilisées pour l'entraînement et les tests (GraphData).
        """
        self.args = args
        self.net = net
        self.feat_dim = G_data.feat_dim  # Dimension des caractéristiques des graphes
        self.fold_idx = G_data.fold_idx  # Indice du pli de validation croisée
        # Initialisation des données d'entraînement et de test
        self.init(args, G_data.train_gs, G_data.test_gs)

    def init(self, args, train_gs, test_gs):
        """
        Initialisation des données d'entraînement et de test, ainsi que de l'optimiseur.

        Arguments:
            args : Arguments contenant les configurations comme le batch size et le taux d'apprentissage.
            train_gs : List des graphes d'entraînement.
            test_gs : List des graphes de test.
        """
        print('#train: %d, #test: %d' % (len(train_gs), len(test_gs)))  # Affiche la taille des datasets
        # Crée des objets GraphData pour les ensembles d'entraînement et de test
        train_data = GraphData(train_gs, self.feat_dim)
        test_data = GraphData(test_gs, self.feat_dim)
        # Charge les données d'entraînement et de test avec les tailles de batchs spécifiées
        self.train_d = train_data.loader(self.args.batch, True)
        self.test_d = test_data.loader(self.args.batch, False)
        # Initialisation de l'optimiseur Adam avec un taux d'apprentissage
        self.optimizer = tf.optimizers.Adam(learning_rate=self.args.lr)

    def to_device(self, gs):
        """
        Déplace les graphes et leurs caractéristiques sur le GPU si disponible.

        Arguments:
            gs : Le graphe ou une liste de graphes à déplacer sur le périphérique (GPU ou CPU).
        
        Retourne:
            Le graphe déplacé vers le périphérique (GPU ou CPU).
        """
        if tf.config.list_physical_devices('GPU'):  # Vérifie si un GPU est disponible
            # Si les données sont sous forme de liste de graphes, les déplacer individuellement
            if isinstance(gs, list):
                return [g.cuda() for g in gs]
            return gs.cuda()  # Déplace un seul graphe vers le GPU
        return gs  # Si pas de GPU, retourne les données sur le CPU

    def run_epoch(self, epoch, data, model, optimizer):
        """
        Exécute un epoch (entraînement ou évaluation) du modèle.

        Arguments:
            epoch : L'indice de l'epoch actuel.
            data : Les données à utiliser pour l'epoch (soit d'entraînement soit de test).
            model : Le modèle de réseau de neurones à utiliser.
            optimizer : L'optimiseur pour mettre à jour les poids du modèle pendant l'entraînement.

        Retourne:
            avg_loss : La perte moyenne pour cet epoch.
            avg_acc : La précision moyenne pour cet epoch.
        """
        losses, accs, n_samples = [], [], 0
        for batch in tqdm(data, desc=str(epoch), unit='b'):  # Utilisation de tqdm pour une barre de progression
            cur_len, gs, hs, ys = batch
            # Déplace les graphes, caractéristiques et labels sur le périphérique (GPU/CPU)
            gs, hs, ys = map(self.to_device, [gs, hs, ys])
            
            if optimizer is None:  # Mode évaluation (pas de mise à jour des gradients)
                loss, acc = model(gs, hs, ys)  # Calcule la perte et la précision sans calcul de gradients
            else:
                # Mode entraînement, calcul des gradients pour optimiser les poids
                with tf.GradientTape() as tape:
                    loss, acc = model(gs, hs, ys)
                
                # Calcule les gradients par rapport aux variables d'entraînement
                gradients = tape.gradient(loss, model.trainable_variables)
                # Applique les gradients sur les variables du modèle pour les mettre à jour
                optimizer.apply_gradients(zip(gradients, model.trainable_variables))

            # Ajoute la perte et la précision pondérées par le nombre d'échantillons dans le batch
            losses.append(loss * cur_len)
            accs.append(acc * cur_len)
            n_samples += cur_len

        # Calcule la perte et la précision moyennes pour l'epoch
        avg_loss, avg_acc = sum(losses) / n_samples, sum(accs) / n_samples
        return avg_loss, avg_acc

    def train(self):
        """
        Exécute l'entraînement du modèle pendant un certain nombre d'epochs.
        Effectue aussi l'évaluation à la fin de chaque epoch.
        """
        max_acc = 0.0  # Initialisation de la précision maximale
        train_str = 'Train epoch %d: loss %.5f acc %.5f'  # Format pour afficher les résultats d'entraînement
        test_str = 'Test epoch %d: loss %.5f acc %.5f max %.5f'  # Format pour afficher les résultats de test
        line_str = '%d:\t%.5f\n'  # Format pour sauvegarder la précision maximale

        for e_id in range(self.args.num_epochs):  # Boucle sur chaque epoch
            # Mode entraînement
            loss, acc = self.run_epoch(e_id, self.train_d, self.net, self.optimizer)
            print(train_str % (e_id, loss, acc))

            # Mode évaluation (mode test), aucun calcul de gradient
            loss, acc = self.run_epoch(e_id, self.test_d, self.net, None)
            max_acc = max(max_acc, acc)  # Mise à jour de la meilleure précision
            print(test_str % (e_id, loss, acc, max_acc))

        # Sauvegarde de la meilleure précision après l'entraînement
        with open(self.args.acc_file, 'a+') as f:
            f.write(line_str % (self.fold_idx, max_acc))
