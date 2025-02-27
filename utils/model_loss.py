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
    e = np.exp(scores)
    sum_s = np.array([np.sum(e, axis=1)])
    s = e / sum_s.T
    softmax_output = s

    # dScores
    one_hot_t = np.zeros(s.shape)
    one_hot_t[np.arange(len(one_hot_t)), t] = 1
    dScores = s - one_hot_t
    dScores /= N

    # Loss
    b0 = model_params["L0"]["b"]
    b1 = model_params["L1"]["b"]
    W0 = model_params["L0"]["W"]
    W1 = model_params["L1"]["W"]
    W = np.array([])
    b = np.array([])
    W = np.append(W, W0)
    W = np.append(W, W1)
    b = np.append(b, b0)
    b = np.append(b, b1)
    norm = np.linalg.norm(W) ** 2 + np.linalg.norm(b) ** 2

    loss = s[np.arange(len(s)), t]
    loss = np.log(loss)
    loss = (-1 / N) * np.sum(loss) + (0.5 * reg * norm)

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
