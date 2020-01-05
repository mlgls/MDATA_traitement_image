import cv2
import numpy as np

#  Mettre l'image bgr.png dans le dossier courant

img = cv2.imread('bgr.png')
blues, greens, reds = cv2.split(img)
cv2.imshow('Bleu', blues)
cv2.imshow('Vert', greens)
cv2.imshow('Rouge', reds)

img_shape = img.shape

#  on crée une nouvelle image (vide) de la même taille que bgr.png
B = np.zeros(shape=img_shape, dtype=np.uint8)
#  on met dans cette image uniquement le canal bleu
B[:, :, 0] = blues
cv2.imshow("B", B)

G = np.zeros(shape=img_shape, dtype=np.uint8)
G[:, :, 1] = greens
cv2.imshow("G", G)

R = np.zeros(shape=img_shape, dtype=np.uint8)
R[:, :, 2] = reds
cv2.imshow("R", R)

cv2.imshow("B+G", B + G)
cv2.imshow("G+R", G + R)
cv2.imshow("R+B", R + B)

key = cv2.waitKey(0)
if key == ord('q'):
    cv2.destroyAllWindows()
