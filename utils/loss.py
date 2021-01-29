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
    
    #print("X shape : ", X.shape)
    #print("W shape : ", W.shape)
    #print("y shape : ", y.shape)
    
    for i in range(N):
        
        #dot product
        x = np.array([X[i]])
        
        y_w = np.dot(x, W)
        #calcul du softmax + normalisation
        y_exp = np.exp(y_w)
        S = y_exp / np.sum(y_exp)
        #calcul de la loss
        loss_tmp = -np.log(S[0, y[i]])
        loss += loss_tmp
        
        #calcul du gradient pour une donnée
        t = np.zeros((1, C))
        t = t[0]
        t[y[i]] = 1
        
        S_t = S - t
        S_t = np.array(S_t).T
        grad_tmp = np.dot(S_t, x).T + reg*W
        dW += grad_tmp

    loss /= N
    dW /= N
    
    loss += 0.5*reg*(np.linalg.norm(W)**2)

    #print(dW)
    
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
    
    #dot product
    y_w = np.dot(X,W)
    
    #calcul du softmax vectorisé
    y_w = np.exp(y_w) # Lors du Softmax, des nombres grands peuvent exploser la limite. Il faut alors prendre la valeur maximale et soustraire toutes les valeurs par cette dernière. Puis faire l'exponentiel.
    y_sum = np.array([np.sum(y_w,1)])
    y_sum = y_sum.T
    S = y_w / y_sum
   
    #calcul de la cross-entropy loss vectorisé
    predictions = S[np.arange(len(S)), y]
    loss = -np.sum(np.log(predictions))
    loss /= N
    loss += 0.5*reg*(np.linalg.norm(W)**2)
    
    #calcul du gradient vectorisé
    t = np.zeros((N,C))
    t[np.arange(len(t)), y] = 1 #creation de la matrice des 1-hot-vectors
    S_t = S - t
    grad = np.dot(S_t.T, X).T + reg*W
    
    dW = grad / N
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
    P = np.dot(X, W)
    T = np.dot(X, W[:, y])
    print("X:", np.shape(X))
    print("P:", np.shape(P))
    print("W:", np.shape(W))
    print("T: ", np.shape(W[:, y][:, 0]))
    print("T Values:", W[:, y][:, 0])
    
    

    return loss, dW
