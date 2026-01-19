import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from numba import njit, objmode # Ajout pour l'accélération par GEMINI

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



######Réalisé par MOI-MÊME######
@njit
def initialisation(Hi, Li, N_filtre, Tf, slide, N_classe):
    filtre = np.random.uniform(-1, 1, (N_filtre, Tf, Tf))*0.1
    b_filtre = np.zeros((1, N_filtre))

    Hs = (Hi - Tf + 1) // slide
    Ls = (Li - Tf + 1) // slide

    w = np.random.uniform(-1, 1, (N_filtre*Hs*Ls, N_classe))*0.01
    b = np.zeros((1, N_classe))

    return filtre, b_filtre, w, b

@njit
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
        return np.zeros((N_s, 0, 0))
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

    return sortie

@njit
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
        return np.zeros((N, 0, 0))
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


@njit
def flatten(V):
    #V : volume de dimension N*H*L
    #V : (N, H, L)
    #sortie : (1, N*H*L)

    return V.reshape(1, -1)

@njit
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

@njit
def préActivation(x, w, b):
    #x : entrée (1, E)
    #w : poids (E, N)
    #b : biais (1, N)
    #sortie : (1, N)

    return np.dot(x, w) + b

@njit
def Relu(z):
    #N_s : nombre de neurone dans la couche s
    #H : hauteur
    #L : largeur
    #z : (N_s, H, L)
    #sortie : (N_s, H, L)

    return np.maximum(0, z)

@njit
def softMax(z):
    #N_s : nombre de neurone dans la couche s
    #H : hauteur
    #L : largeur
    #z : (1, N_s)
    #sortie : (1, N_s)
    ez = np.exp(z - np.max(z))
    Sez = np.sum(ez)

    return ez/Sez

@njit
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

@njit
def t_vecteur(t):
    t_v = np.zeros((1,10))
    t_v[0][t] = 1
    return t_v

@njit
def dmaxPool_dyflatten(y_filtre_n, y_flatten_n, slide, préGrad):
    #y_filtre_n : (Hi - Tf + 1, Li - Tf + 1)
    #y_flatten_n : ((Hi - Tf + 1) // slide * (Li - Tf + 1) // slide)
    #préGrad : (1, (Hi - Tf + 1) // slide * (Li - Tf + 1) // slide)

    taille_totale = y_filtre_n.shape[0]*y_filtre_n.shape[1]
    sortie = np.zeros(taille_totale)

    grad_flat = préGrad.flatten()

    d0 = y_flatten_n.shape[0]

    for i in range (d0):

        i_flatten = i // (y_filtre_n.shape[1]//slide)
        j_flatten = i - i_flatten*(y_filtre_n.shape[1]//slide)

        for l in range(slide):
            for m in range(slide):
                idx_i = slide*i_flatten + l
                idx_j = slide*j_flatten + m
                if y_filtre_n[idx_i][idx_j] == y_flatten_n[i]:
                    indice = slide*j_flatten + m + (slide*i_flatten + l)*y_filtre_n.shape[1]
                    sortie[indice] += grad_flat[i]
    return sortie.reshape(1, -1)

@njit
def dReLu_dzfiltre(z_filtre_n, préGrad):
    #z_filtre_n : (Hi - Tf + 1, Li - Tf + 1)
    #préGrad : (1, (Hi - Tf + 1)*(Li - Tf + 1))
    #sortie : (1, (Hi - Tf + 1)*(Li - Tf + 1))

    sortie = np.zeros((1, z_filtre_n.shape[0]*z_filtre_n.shape[1]))
    grad_flat = préGrad.flatten()

    for i in range(z_filtre_n.shape[0]):
        for j in range(z_filtre_n.shape[1]):
            indice = j + i*z_filtre_n.shape[1]
            if z_filtre_n[i][j] >= 0:
                sortie[0][indice] += grad_flat[indice]
    return sortie

@njit
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

@njit
def batch_training(images, t, filtre, b_filtre, slide, w, b, step, n_epochs):
    #images : (Ni, Hi, Li)
    #t (target) : (Ni)
    #filtre : (N, Tf, Tf)
    #b_filtre : (1, N)
    #slide pour le maxPool
    #N_flatten  = N*((Hi - Tf + 1) // slide)*((Li - Tf + 1) // slide)
    #w : (N_flatten, N_s)
    #b : (1, N_s)
    #n_epochs : Nombre de fois qu'on parcourt toute la base

    N_batch = 100
    iteration = images.shape[0] // N_batch
    N_filtre = filtre.shape[0]

    w_train = w.copy()
    b_train = b.copy()

    w_filtre_train = filtre.copy()
    b_filtre_train = b_filtre.copy()

    for epoch in range(n_epochs):

        # Compteurs pour afficher les stats tous les 10 batchs
        running_correct = 0
        running_loss = 0.0
        running_samples = 0

        for l in range(iteration):
            indices_source = np.arange(images.shape[0])
            batch_indices = np.random.choice(indices_source, size=N_batch, replace=False)

            images_batch = images[batch_indices]
            t_batch = t[batch_indices]

            Grad_w = np.zeros(w.shape)
            Grad_b = np.zeros(b.shape)

            Grad_w_filtre = np.zeros(filtre.shape)
            Grad_b_filtre = np.zeros(b_filtre.shape)

            for m in range(N_batch):
                z_filtre, y_filtre, y_filtre_maxPool, y_flatten, z, y = forward(images_batch[m], w_filtre_train, b_filtre_train, slide, w_train, b_train)
                tv = t_vecteur(t_batch[m])

                # --- CALCUL ACCURACY & LOSS (Mise à jour running) ---
                loss_val = -np.sum(tv * np.log(y + 1e-9))
                running_loss += loss_val

                prediction = np.argmax(y)
                if prediction == t_batch[m]:
                    running_correct += 1
                running_samples += 1
                # ------------------------------

                delta = y - tv
                Grad_w += np.dot(y_flatten.T, delta)
                Grad_b += delta


                for n in range (N_filtre):
                    number = y_filtre_maxPool.shape[1]*y_filtre_maxPool.shape[2]

                    w_slice = w_train[n*number: (n + 1)*number]
                    préGrad = np.dot(delta, w_slice.T)

                    y_filtre_n = y_filtre[n]
                    y_flatten_slice = y_flatten[0][n*number: (n + 1)*number]
                    préGrad = dmaxPool_dyflatten(y_filtre_n, y_flatten_slice, slide, préGrad)

                    z_filtre_n = z_filtre[n]
                    préGrad = dReLu_dzfiltre(z_filtre_n, préGrad)

                    #Numba ne traite pas la 3D donc Grad_w_filtre[n] += np.dot(dconv2D_dfiltre_n(images_batch[m], filtre[n]), préGrad[0]) est remplacé

                    #Remplacement
                    # 1. On calcule la matrice 3D
                    mat_3d = dconv2D_dfiltre_n(images_batch[m], filtre[n])
                    # 2. On aplatit temporairement les deux premières dimensions
                    mat_reshaped = mat_3d.reshape(-1, mat_3d.shape[2])
                    # 3. On fait le dot, puis on remet sous forme carrée
                    dot_res = np.dot(mat_reshaped, préGrad[0])
                    Grad_w_filtre[n] += dot_res.reshape(filtre.shape[1], filtre.shape[2])
                    #Remplacement

                    Grad_b_filtre[0][n] += préGrad[0].sum()

            w_train -= step/N_batch*Grad_w
            b_train -= step/N_batch*Grad_b

            w_filtre_train -= step/N_batch*Grad_w_filtre
            b_filtre_train -= step/N_batch*Grad_b_filtre

            #Affichage
            if (l + 1) % 10 == 0:
                avg_loss = running_loss / running_samples
                avg_acc = running_correct / running_samples

                with objmode():
                    print("Epoch", epoch + 1, "| Batch", l + 1, "/", iteration, "| Loss:", avg_loss, "| Accuracy:", avg_acc, flush=True)

                running_correct = 0
                running_loss = 0.0
                running_samples = 0

    return w_filtre_train, b_filtre_train, w_train, b_train


@njit
def calcul_accuracy_test(X_test, y_test, filtre, b_filtre, slide, w, b):
    correct = 0
    total = X_test.shape[0]

    for i in range(total):
        _, _, _, _, _, y_pred = forward(X_test[i], filtre, b_filtre, slide, w, b)

        if np.argmax(y_pred) == y_test[i]:
            correct += 1

    return correct / total * 100


# ==========================================
# LANCEMENT DU PROGRAMME
# ==========================================

Hi, Li = X_train.shape[1], X_train.shape[2]
N_filtre = 10
Tf = 3
Slide = 2
N_classe = 10

print("Initialisation...")
filtre, b_filtre, w, b = initialisation(Hi, Li, N_filtre, Tf, Slide, N_classe)

print("Entraînement...")
step = 0.05
n_epochs = 7 # On fait 7 tours complets de la base de données

# Premier appel un peu lent (compilation), les suivants seront instantanés
w_f_fin, b_f_fin, w_fin, b_fin = batch_training(X_train, y_train, filtre, b_filtre, Slide, w, b, step, n_epochs)

print("Terminé !")

print("Test sur la base de donnée MNIST (qui n'est pas la base de donnée de l'entrainement)...")
print("Accuracy sur la base de test MNIST:")
print(calcul_accuracy_test(X_test, y_test, w_f_fin, b_f_fin, Slide, w_fin, b_fin))

######Réalisé par MOI-MÊME######






# ==========================================
# 7. INTERFACE GRAPHIQUE (PAINT) (FAIT PAR GEMINI)
# ==========================================
import tkinter as tk

def lancer_interface(filtre, b_filtre, slide, w, b):
    print("Ouverture de la fenêtre de dessin pour essayer l'IA. Veuillez dessiner le plus proche possible que la base MNIST (centré...) car l'IA n'a été entrainé que sur ce type de numéros.")

    # Configuration
    ZOOM = 10  # On affiche en 280x280 pour que ce soit utilisable
    CANVAS_SIZE = 28 * ZOOM

    # La grille mémoire (l'image réelle vue par l'IA)
    # Initialisée à 0 (Noir)
    grid_data = np.zeros((28, 28))

    # Création de la fenêtre
    root = tk.Tk()
    root.title("Test CNN - Dessinez un chiffre")

    # Zone de dessin
    canvas = tk.Canvas(root, width=CANVAS_SIZE, height=CANVAS_SIZE, bg='black', cursor="cross")
    canvas.pack(pady=10, padx=10)

    # --- FONCTIONS ---

    def paint(event):
        # Coordonnées souris -> Coordonnées matrice 28x28
        col = event.x // ZOOM
        row = event.y // ZOOM

        # On dessine un "Gros" trait (3x3 pixels) pour imiter l'épaisseur MNIST
        # Sinon le trait est trop fin et les filtres ne voient rien
        for i in range(-1, 2):
            for j in range(-1, 2):
                r, c = row + i, col + j
                if 0 <= r < 28 and 0 <= c < 28:
                    # 1. On met à jour la mémoire (IA)
                    # On met une intensité de 1.0 (Blanc pur) ou on accumule
                    grid_data[r][c] = min(1.0, grid_data[r][c] + 0.5)

                    # 2. On met à jour l'écran (Visuel)
                    x1 = c * ZOOM
                    y1 = r * ZOOM
                    canvas.create_rectangle(x1, y1, x1+ZOOM, y1+ZOOM, fill='white', outline='white')

    def deviner():
        # On lance la propagation avant (Forward)
        # Numba accepte sans problème le tableau numpy venant de Python
        _, _, _, _, _, y_out = forward(grid_data, filtre, b_filtre, slide, w, b)

        prediction = np.argmax(y_out)
        proba = y_out[0][prediction] * 100

        # Mise à jour de l'affichage
        lbl_result.config(text=f"C'est un {prediction} !\n(Confiance : {proba:.1f}%)", fg="green")

        # Debug console
        print(f"Prédiction : {prediction} | Proba : {np.round(y_out, 2)}")

    def effacer():
        canvas.delete("all")
        grid_data[:] = 0 # Remise à zéro de la matrice
        lbl_result.config(text="Dessinez...", fg="black")

    # --- WIDGETS ---

    # Liaison souris (Clic gauche maintenu)
    canvas.bind("<B1-Motion>", paint)
    # Pour dessiner juste en cliquant sans bouger
    canvas.bind("<Button-1>", paint)

    btn_frame = tk.Frame(root)
    btn_frame.pack(fill='x', padx=10)

    btn_predict = tk.Button(btn_frame, text="DEVINER", command=deviner, bg="#4CAF50", fg="white", font=("Arial", 12, "bold"))
    btn_predict.pack(side='left', expand=True, fill='x', padx=5)

    btn_clear = tk.Button(btn_frame, text="EFFACER", command=effacer, bg="#f44336", fg="white", font=("Arial", 12, "bold"))
    btn_clear.pack(side='right', expand=True, fill='x', padx=5)

    lbl_result = tk.Label(root, text="Dessinez au centre...", font=("Arial", 16))
    lbl_result.pack(pady=10)

    root.mainloop()

# Lancement final
lancer_interface(w_f_fin, b_f_fin, Slide, w_fin, b_fin)