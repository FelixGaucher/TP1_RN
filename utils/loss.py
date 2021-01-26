import numpy as np


def softmax_ce_naive_forward_backward(X, W, y, reg):
    """Implémentation naive qui calcule la propagation avant, puis la
       propagation arrière pour finalement retourner la perte entropie croisée
       (ce) + une régularisation L2 et le gradient des poids. Utilise une 
       activation softmax en sortie.
       
       NOTE : la fonction codée est : EntropieCroisée + 0.5*reg*||W||^2
       
       N'oubliez pas le 0.5!

    Inputs:
    - X: Numpy array, shape (N, D). N représente le nombre d'exemple d'entrainement
        dans X, et D représente la dimension des exemples de X.
    - W: Numpy array, shape (D, C)
    - y: Numpy array, shape (N,). y[i] = c veut dire que X[i] appartient à la
        classe c, 0 <= c < C
    - reg: float. Terme de regularisation L2

    Outputs:
    - loss: float. Perte du classifieur linéaire softmax
    - dW: Numpy array, shape (D, C). Gradients des poids W
    """

    N = X.shape[0]
    C = W.shape[1]

    loss = 0
    dW = np.zeros(W.shape)

    ### TODO ###
    # Ajouter code ici #

    return loss, dW


def softmax_ce_forward_backward(X, W, y, reg):
    """Implémentation vectorisée qui calcule la propagation avant, puis la
       propagation arrière pour finalement retourner la perte entropie croisée
       (ce) et le gradient des poids. Utilise une activation softmax en sortie.
        
       NOTE : la fonction codée est : EntropieCroisée + 0.5*reg*||W||^2      
       N'oubliez pas le 0.5!

    Inputs:
    - X: Numpy array, shape (N, D). N représente le nombre d'exemples d'entrainement
        dans X, et D représente la dimension des exemples de X.
    - W: Numpy array, shape (D, C)
    - y: Numpy array, shape (N,). y[i] = c veut dire que X[i] appartient à la
        classe c, 0 <= c < C
    - reg: float. Terme de regularisation L2

    Outputs:
    - loss: float. Perte du classifieur linéaire softmax
    - dW: Numpy array, shape (D, C). Gradients des poids W
    """
    N = X.shape[0]
    C = W.shape[1]
    loss = 0.0
    dW = np.zeros(W.shape)
    ### TODO ###
    # Ajouter code ici #
    
    return loss, dW


def hinge_naive_forward_backward(X, W, y, reg):
    """Implémentation naive calculant la propagation avant, puis la
       propagation arrière, pour finalement retourner la perte hinge et le
       gradient des poids.
       
       NOTE : la fonction codée est : Hinge + 0.5*reg*||W||^2
       N'oubliez pas le 0.5!

    Inputs:
    - X: Numpy array, shape (N, D)
    - W: Numpy array, shape (D, C)
    - y: Numpy array, shape (N,). y[i] = c veut dire que X[i] appartient à la
         classe c, 0 <= c < C
    - reg: float. Terme de regularisation L2

    Outputs:
    - loss: float. Perte du classifieur linéaire hinge
    - dW: Numpy array, shape (D, C). Gradients des poids W
    """
    loss = 0.0
    dW = np.zeros(W.shape)

    ### TODO ###
    # Ajouter code ici #
    for i in range(np.size(y)):
        predict = np.argmax(np.dot(W.T, X[i])) # Prédiction de la classe.
        loss += max(0, 1 + np.dot(W.T[predict], X[i]) - np.dot(W.T[y[i]], X[i])) # Calcul de la Hinge pour une donnée.
        
        # Calcul du gradient
        dW.T[predict] += X[i]
        dW.T[y[i]] -= X[i]

    loss /= np.size(y)
    loss += 0.5*reg*(np.linalg.norm(W)**2) # Ajout du terme de régularisation
    dW /= np.size(y)
    
    return loss, dW


def hinge_forward_backward(X, W, y, reg):
    """Implémentation vectorisée calculant la propagation avant, puis la
       propagation arrière, pour finalement retourner la perte hinge et le
       gradient des poids.

       NOTE : la fonction codée est : Hinge + 0.5*reg*||W||^2
       N'oubliez pas le 0.5!
       
    Inputs:
    - X: Numpy array, shape (N, D)
    - W: Numpy array, shape (D, C)
    - y: Numpy array, shape (N,). y[i] = c veut dire que X[i] appartient à la
         classe c, 0 <= c < C
    - reg: float. Terme de regularisation L2

    Outputs:
    - loss: float. Perte du classifieur linéaire hinge
    - dW: Numpy array, shape (D, C). Gradients des poids W
    """
    loss = 0.0
    dW = np.zeros(W.shape)

    ### TODO ###
    # Ajouter code ici #

    return loss, dW
