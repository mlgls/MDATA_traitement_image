import cv2
import glob
import numpy as np
from operator import itemgetter
from collections import Counter
import pandas as pd
from sklearn.metrics import accuracy_score

# choix du dataset : (les calculs sont beaucoup plus longs pour le dataset2)
dataset = 1  # ou 2
# choix du nombre de voisins à garder pour les images les plus proches
nb_img = 2 # dataset1 : nb=7 pour 100% de precision

if dataset == 1:
    # DATA1
    # Images test
    label_test_data = [1, 3, 5, 7, 9, 14]  # label des 6 images test du dataset 1
    path = glob.glob("Data1/Test/*.png")  # recupere les images du dossier Test
    img_test = []  # liste des vecteurs images test
    for img in path:
        n = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
        img_test.append(n)
    # Images du dataset
    path = glob.glob("Data1/References/*.png")
    # liste des vecteurs images
    img_data = []
    for img in path:
        n = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
        img_data.append(n)
    data_label = ([1] * 20) + ([3] * 20) + ([5] * 20) + ([7] * 20) + ([9] * 20) + (
                [14] * 20)  # label des images du dataset
elif dataset == 2:
    # DATA 2
    # Images test
    label_test_data = ([1] * 20) + ([3] * 20) + ([5] * 20) + ([7] * 20) + ([9] * 20) + (
                [14] * 20)  # label des 6 images test du dataset 1
    path = glob.glob("Data2/Tests/*.png")  # recupere les images du dossier Test
    img_test = []  # liste des vecteurs images test
    for img in path:
        n = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
        img_test.append(n)
    # Images du dataset
    path = glob.glob("Data2/References/*.png")
    # liste des vecteurs images
    img_data = []
    for img in path:
        n = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
        img_data.append(n)
    data_label = ([1] * 20) + ([3] * 20) + ([5] * 20) + ([7] * 20) + ([9] * 20) + (
                [14] * 20)  # label des images du dataset

predicted = []

for index, img in enumerate(img_test):
    # === Brute-Force Matching with SIFT Descriptors + ratio ===
    # Initiate SIFT detector
    sift = cv2.xfeatures2d.SIFT_create()
    # Find descriptors and keypoints for each image
    dist_moy = []
    best = []
    indice = -1

    # pour chaque image du dataset, on va calculer la distance moyenne des meilleurs matchs SIFT entre l'image et l'image test
    for image in img_data:
        indice += 1
        kp_test, des_test = sift.detectAndCompute(img, None)  # kp:90, des:90*128
        kp_data, des_data = sift.detectAndCompute(image, None)
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des_test, des_data, k=2)
        best_for_img = []
        dist = []
        for m, n in matches:  # on applique le critere de David Lowe pour garder les meilleurs matchs
            if m.distance < .75 * n.distance:
                best_for_img.append(m)
                dist.append(m.distance)
        best.append(best_for_img)
        if dist is not 0:
            # on stocke la distance moyenne des match et le label associé à l'image
            dist_moy.append((np.mean(dist), data_label[indice]))

    # on supprime tous les "nan" de la liste
    data = list(filter(lambda x: not np.isnan(x[0]), dist_moy))
    # on trie dans l'ordre croissant selon la distance
    res = sorted(data, key=itemgetter(0))
    label_knn = res[:nb_img]  # on prend les k images les plus proches
    label = Counter([(item[1]) for item in label_knn]).most_common()[0][0]  # on prend le label qui apparait le plus souvent
    print(label_knn)
    print("Catégorie de l'image :", label_test_data[index])
    print("Prediction : ", label)
    predicted.append(label)

# Matrice de confusion
y_actu = pd.Series(predicted, name='Actual')
y_pred = pd.Series(label_test_data, name='Predicted')
confusion_matrix = pd.crosstab(y_actu, y_pred, rownames=['Actual'], colnames=['Predicted'], margins=True)
print(" Matrice de confusion:\n", confusion_matrix)

# taux de reconnaissance obtenu
accuracy = accuracy_score(y_actu, y_pred)
print("Précision du modèle :", accuracy * 100," %")
