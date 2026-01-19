import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

######Données préparées par GEMINI######
# 1. Chargement
print("Chargement...")
mnist = fetch_openml('mnist_784', version=1, as_frame=False)
X_raw, y_raw = mnist.data, mnist.target

# 2. Conversion labels et Normalisation
y = y_raw.astype(np.uint8)
X = X_raw / 255.0  # Passage de 0-255 à 0-1

# 3. LE RESHAPE DÉFINITIF
# On transforme la liste plate en une pile d'images carrées
# -1 = "tout le monde", 28 = hauteur, 28 = largeur
X = X.reshape(-1, 28, 28)

# 4. Séparation Train / Test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=10000, random_state=42, stratify=y
)
######Données préparées par GEMINI######




def initialisation(Hi, Li, N_filtre, Tf, slide, N_classe):
    filtre = np.random.uniform(-1, 1, (N_filtre, Tf, Tf))*0.1
    b_filtre = np.zeros((1, N_filtre))

    Hs = (Hi - Tf + 1) // slide
    Ls = (Li - Tf + 1) // slide

    w = np.random.uniform(-1, 1, (N_filtre*Hs*Ls, N_classe))*0.01
    b = np.zeros((1, N_classe))

    return filtre, b_filtre, w, b

def conv2D(image, filtre):
    #Hi : hauteur image
    #Li : largeur image
    #Tf : taille filtre
    #N_s : nombre de neurone de la couche s

    #image : (Hi, Li)
    #filtre : (N_s, Tf, Tf)
    #sortie : (N_s, Hs = Hi - Tf + 1, Ls = Li - Tf + 1)

    Hi, Li = image.shape
    N_s = filtre.shape[0]
    Tf = filtre.shape[1]

    if (Hi < Tf or Li < Tf or Tf != filtre.shape[2] or N_s <= 0):
        print("Impossible de convoluer")
        return np.array([])
    else :
        Hs = Hi - Tf + 1
        Ls = Li - Tf + 1
        sortie = np.zeros((N_s, Hs, Ls))

        for k in range (N_s):
            for i in range (Hs):
                for j in range (Ls):
                    conv = 0
                    for l in range(Tf):
                        for m in range(Tf):
                            conv += image[i + l][j + m]*filtre[k][l][m]
                    sortie[k][i][j] = conv

    return np.array(sortie)


def maxPool(y_filtre, slide):
    #N : nombre d'image

    #Hi : hauteur de l'image
    #Li : largeur de l'image
    #y_filtre : (N, Hi, Li)

    #Hs : hauteur de l'image de sortie
    #Ls : largeur de l'image de sortie
    #sortie : (N, Hs = Hi // slide, Ls = Li // slide)

    N, Hi, Li = y_filtre.shape
    if slide < 0 or N < 0:
        return np.array([])
    else:
        Hs = Hi // slide
        Ls = Li // slide

        sortie = np.zeros((N, Hs, Ls))
        for k in range(N):
            for i in range(Hs):
                for j in range(Ls):
                    max = y_filtre[k][slide*i][slide*j]
                    for l in range(slide):
                        for m in range(slide):
                            if max < y_filtre[k][slide*i + l][slide*j + m] :
                                max = y_filtre[k][slide*i + l][slide*j + m]
                    sortie[k][i][j] = max

        return sortie


def flatten(V):
    #V : volume de dimension N*H*L
    #V : (N, H, L)
    #sortie : (1, N*H*L)

    return V.reshape(1, -1)

def préActivationFiltre(image, filtre, b):
    #Hi : hauteur image
    #Li : largeur image
    #Tf : taille filtre
    #N_s : nombre de neurone de la couche s

    #image : (Hi, Li)
    #filtre : (N_s, Tf, Tf)
    #b : (1, N_s)

    #sortie : (N_s, Hs = Hi - Tf + 1, Ls = Li - Tf + 1)

    N_s = filtre.shape[0]
    sortie = conv2D(image, filtre)

    for k in range(N_s):
        sortie[k] = sortie[k] + b[0][k]

    return sortie

def préActivation(x, w, b):
    #x : entrée (1, E)
    #w : poids (E, N)
    #b : biais (1, N)
    #sortie : (1, N)

    return np.dot(x, w) + b

def Relu(z):
    #N_s : nombre de neurone dans la couche s
    #H : hauteur
    #L : largeur
    #z : (N_s, H, L)
    #sortie : (N_s, H, L)

    return np.maximum(0, z)

def softMax(z):
    #N_s : nombre de neurone dans la couche s
    #H : hauteur
    #L : largeur
    #z : (1, N_s)
    #sortie : (1, N_s)
    ez = np.exp(z - np.max(z))
    Sez = np.sum(ez)

    return ez/Sez

def forward(image, filtre, b_filtre, slide, w, b):
    #image : (Hi, Li)
    #filtre : (N, Tf, Tf)
    #b_filtre : (1, N)
    #slide pour le maxPool
    #N_flatten  = N*((Hi - Tf + 1) // slide)*((Li - Tf + 1) // slide)
    #w : (N_flatten, N_s)
    #b : (1, N_s)

    #z_filtre : (N, Hi - Tf + 1, Li - Tf + 1)
    #y_filtre : (N, Hi - Tf + 1, Li - Tf + 1)
    #y_filtre_maxPool : (N, (Hi - Tf + 1) // slide, (Li - Tf + 1) // slide)
    #y_flatten : (1, N_flatten)
    #z : (1, N_s)
    #y : (1, N_s)

    z_filtre = préActivationFiltre(image, filtre, b_filtre)
    y_filtre = Relu(z_filtre)
    y_filtre_maxPool = maxPool(y_filtre, slide)
    y_flatten = flatten(y_filtre_maxPool)

    z = préActivation(y_flatten, w, b)
    y = softMax(z)

    return z_filtre, y_filtre, y_filtre_maxPool, y_flatten, z, y

def t_vecteur(t):
    t_v = np.zeros((1,10))
    t_v[0][t] = 1
    return t_v

def dmaxPool_dyflatten(y_filtre_n, y_flatten_n, slide, préGrad):
    #y_filtre_n : (Hi - Tf + 1, Li - Tf + 1)
    #y_flatten_n : ((Hi - Tf + 1) // slide * (Li - Tf + 1) // slide)
    #préGrad : (1, (Hi - Tf + 1) // slide * (Li - Tf + 1) // slide)

    sortie = np.zeros((1,y_filtre_n.shape[0]*y_filtre_n.shape[1]))

    d0 = y_flatten_n.shape[0]

    for i in range (d0):

        i_flatten = i // (y_filtre_n.shape[1]//slide)
        j_flatten = i - i_flatten*(y_filtre_n.shape[1]//slide)

        for l in range(slide):
            for m in range(slide):
                if y_filtre_n[slide*i_flatten + l][slide*j_flatten + m] == y_flatten_n[i]:
                    indice = slide*j_flatten + m + (slide*i_flatten + l)*y_filtre_n.shape[1]
                    sortie[0][indice] += préGrad[0][i]
    return sortie

def dReLu_dzfiltre(z_filtre_n, préGrad):
    #z_filtre_n : (Hi - Tf + 1, Li - Tf + 1)
    #préGrad : (1, (Hi - Tf + 1)*(Li - Tf + 1))
    #sortie : (1, (Hi - Tf + 1)*(Li - Tf + 1))

    sortie = np.zeros((1, z_filtre_n.shape[0]*z_filtre_n.shape[1]))

    for i in range(z_filtre_n.shape[0]):
        for j in range(z_filtre_n.shape[1]):
            indice = j + i*z_filtre_n.shape[1]
            if z_filtre_n[i][j] >= 0:
                sortie[0][indice] += préGrad[0][indice]
    return sortie

def dconv2D_dfiltre_n(image_m, filtre_n):
    #image_n : (Hi, Li)
    #filtre_n : (Tf, Tf)
    #sortie : (Tf, Tf, Hs*Ls)

    Hi, Li = image_m.shape
    Tf = filtre_n.shape[0]

    Hs = Hi - Tf + 1
    Ls = Li - Tf + 1

    sortie = np.zeros((Tf, Tf, Hs*Ls))

    for i in range(Tf):
        for j in range(Tf):
            for k in range(Hs * Ls):
                i_filtre = k // Ls
                j_filtre = k - i_filtre*Ls
                sortie[i][j][k] = image_m[i_filtre + i][j_filtre + j]

    return sortie


def batch_training(images, t, filtre, b_filtre, slide, w, b, step):
    #images : (Ni, Hi, Li)
    #t (target) : (Ni)
    #filtre : (N, Tf, Tf)
    #b_filtre : (1, N)
    #slide pour le maxPool
    #N_flatten  = N*((Hi - Tf + 1) // slide)*((Li - Tf + 1) // slide)
    #w : (N_flatten, N_s)
    #b : (1, N_s)

    N_batch = 100
    iteration = images.shape[0] // N_batch
    N_filtre = filtre.shape[0]

    w_train = w.copy()
    b_train = b.copy()

    w_filtre_train = filtre.copy()
    b_filtre_train = b_filtre.copy()

    for l in range(iteration):

        batch_indices = np.random.choice(images.shape[0], size=N_batch, replace=False)
        images_batch = images[batch_indices]
        t_batch = t[batch_indices]

        Grad_w = np.zeros(w.shape)
        Grad_b = np.zeros(b.shape)

        Grad_w_filtre = np.zeros(filtre.shape)
        Grad_b_filtre = np.zeros(b_filtre.shape)

        for m in range(N_batch):
            z_filtre, y_filtre, y_filtre_maxPool, y_flatten, z, y = forward(images_batch[m], w_filtre_train, b_filtre_train, slide, w_train, b_train)
            tv = t_vecteur(t_batch[m])

            delta = y - tv
            Grad_w += np.dot(y_flatten.T, delta)
            Grad_b += delta


            for n in range (N_filtre):
                number = y_filtre_maxPool.shape[1]*y_filtre_maxPool.shape[2]
                préGrad = np.dot(delta, w_train[n*number: (n + 1)*number][:].T)

                y_filtre_n = y_filtre[n]
                préGrad = dmaxPool_dyflatten(y_filtre_n, y_flatten[0][n*number: (n + 1)*number], slide, préGrad)

                z_filtre_n = z_filtre[n]
                préGrad = dReLu_dzfiltre(z_filtre_n, préGrad)

                Grad_w_filtre[n] += np.dot(dconv2D_dfiltre_n(images_batch[m], filtre[n]), préGrad[0])
                Grad_b_filtre[0][n] += préGrad[0].sum()

        w_train -= step/N_batch*Grad_w
        b_train -= step/N_batch*Grad_b

        w_filtre_train -= step/N_batch*Grad_w_filtre
        b_filtre_train -= step/N_batch*Grad_b_filtre

    return w_filtre_train, b_filtre_train, w_train, b_train

Hi, Li = X_train.shape[1], X_train.shape[2]
N_filtre = 8
Tf = 3
Slide = 2
N_classe = 10

