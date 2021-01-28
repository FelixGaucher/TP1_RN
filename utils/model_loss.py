import numpy as np


def cross_entropy_loss(scores, t, reg, model_params):
    """Calcule l'entropie croisée multi-classe.

    Arguments:
        scores {ndarray} -- Scores du réseau (sortie de la dernière couche).
                            Shape (N, C)
        t {ndarray} -- Labels attendus pour nos échantillons d'entraînement.
                       Shape (N, )
        reg {float} -- Terme de régulatisation
        model_params {dict} -- Dictionnaire contenant les paramètres de chaque couche
                               du modèle. Voir la méthode "parameters()" de la
                               classe Model.

    Returns:
        loss {float} 
        dScore {ndarray}: dérivée de la loss par rapport aux scores. : Shape (N, C)
        softmax_output {ndarray} : Shape (N, C)
    """
    N = scores.shape[0]
    C = scores.shape[1]
    loss = 0
    dScores = np.zeros(scores.shape)
    softmax_output = np.zeros(scores.shape)
    
    # TODO
    # Ajouter code ici
    
    # Softmax
    # a = np.dot(scores.T, t)
    e = np.exp(scores)
    sum_s = np.sum(e)
    s = e / sum_s
    softmax_output = np.argmax(s)

    # dScores
    print("score : ", scores.shape)
    print("s : ", np.shape(s))
    print("t : ", t)
    one_hot_t = np.zeros(s.shape)
    one_hot_t[:, t] = 1
    dScores = s - t

    # Loss
    b0 = np.array(model_params["L0"]["b"])[:, np.newaxis]
    b1 = np.array(model_params["L1"]["b"])[np.newaxis, :]
    W = np.dot(model_params["L0"]["W"], model_params["L1"]["W"])
    b = np.dot(b0, b1)

    losslog = np.log(s[:, t])
    loss = -1 / N * np.sum(losslog) + reg * (np.linalg.norm(W) ** 2 + np.linalg.norm(b) ** 2)

    return loss, dScores, softmax_output


def hinge_loss(scores, t, reg, model_params):
    """Calcule la loss avec la méthode "hinge loss multi-classe".

    Arguments:
        scores {ndarray} -- Scores du réseau (sortie de la dernière couche).
                            Shape (N, C)
        t {ndarray} -- Labels attendus pour nos échantillons d'entraînement.
                       Shape (N, )
        reg {float} -- Terme de régulatisation
        model_params {dict} -- Dictionnaire contenant l'ensemble des paramètres
                               du modèle. Obtenus avec la méthode parameters()
                               de Model.

    Returns:
        tuple -- Tuple contenant la loss et la dérivée de la loss par rapport
                 aux scores.
    """

    N = scores.shape[0]
    C = scores.shape[1]
    loss = 0
    dScores = np.zeros(scores.shape)
    score_correct_classes = np.zeros(scores.shape)

    # TODO
    # Ajouter code ici
    return loss, dScores, score_correct_classes
