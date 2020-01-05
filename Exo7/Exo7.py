import cv2
import glob
import numpy as np
from operator import itemgetter

# DATA1
# img test
path = glob.glob("Data1/Test/*.png")
# liste des vecteurs images test
img_test = []
for img in path:
    n = cv2.imread(img,cv2.IMREAD_GRAYSCALE)
    img_test.append(n)
# img du dataset
path = glob.glob("Data1/References/*.png")
# liste des vecteurs images
img_data = []
for img in path:
    n = cv2.imread(img,cv2.IMREAD_GRAYSCALE)
    img_data.append(n)

# === Brute-Force Matching with SIFT Descriptors + ratio ===
# Initiate SIFT detector
sift = cv2.xfeatures2d.SIFT_create()
# Find descriptors and keypoints pour chaque image
dist_moy = []
best = []
indice = -1
data_label = ([1]*20)+([3]*20)+([5]*20)+([7]*20)+([9]*20)+([14]*20)
for img in img_data:
    indice += 1
    kp_test, des_test = sift.detectAndCompute(img_test[2],None) # kp:90, des:90*128
    kp_data, des_data = sift.detectAndCompute(img,None)
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des_test, des_data, k=2)
    best_for_img = []
    dist = []
    for m, n in matches:
        if m.distance < .75 * n.distance:
            best_for_img.append(m)
            dist.append(m.distance)
    best.append(best_for_img)
    if dist is not 0:
        dist_moy.append((np.mean(dist),data_label[indice]))

data = dist_moy
nan = float('nan')
for i in data:
    if i[0] is not float:
        data.remove(i)
res = sorted(data,key=itemgetter(0))
label_knn = res[:3]
print(len(label_knn))
print("Prediction : ", label_knn)
