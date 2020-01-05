from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import cv2
import numpy as np

# Mettre dans le dossier courant les images firefox-256.jpg et fleurs.jpg

# choix du traitement : RGB ou HSV (mettre la ligne non voulue en commentaire)
traitement = "rgb"
# traitement = "hsv"

# choix de la photo : firefox ou fleurs (mettre la ligne non voulue en commentaire)
# photo = "firefox-256"
photo = "fleurs"

if photo == "firefox-256":
    image = cv2.imread('firefox-256.jpg')
    img = image
    if traitement == "hsv":
        img = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
elif photo == "fleurs":
    image = cv2.imread('fleurs.jpg')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img = image
    if traitement == "hsv":
        img = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

cv2.imshow('img', image)
image = img.reshape((img.shape[0] * img.shape[1], 3))
image = np.float32(image)  # seul format accepté par opencv kmeans
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
K = 4
ret, label, center = cv2.kmeans(image, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
center = np.uint8(center)
res = center[label.flatten()]
res = res.reshape(img.shape)
if traitement == "hsv":
    res = cv2.cvtColor(res, cv2.COLOR_HSV2RGB)
cv2.imshow('res', res)
cv2.waitKey(0)
cv2.destroyAllWindows()
