import cv2
import matplotlib.pyplot as plt

# Mettre les images image.jpg et template.jpg dans le dossier courant

img_rgb = cv2.imread('image.jpg')
img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)

template = cv2.imread('template.jpg')
template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

w, h = template.shape[::-1]
x, y = img_gray.shape[::-1]

hist_template = cv2.calcHist([template], [0], None, [256], [0, 256])
hist_template = cv2.normalize(hist_template, hist_template)
step_x = int(w / 3)
step_y = int(h / 3)
nb_step_x = int(x / step_x)
nb_step_y = int(y / step_y)
result = []

results = []

for i in range(nb_step_x):

    for j in range(nb_step_y):
        pos_x = i * step_x
        pos_y = j * step_y
        crop_img = img_gray[pos_x:pos_x + w, pos_y:pos_y + h]
        if crop_img.shape == (w, h):
            hist_crop = cv2.calcHist([crop_img], [0], None, [256], [0, 256])
            hist_crop = cv2.normalize(hist_crop, hist_crop)
            # on compare les histogrammes pour trouver les zones qui ressemblent le plus au template
            comp = cv2.compareHist(hist_template, hist_crop,0)
            results.append(comp)

            # si les zones ressemblent fort au template, on retient leur position pour l'indiquer sur l'image
            if comp > 0.906:
                result.append((pos_x, pos_y))
                # cv2.imshow("crop"+str(i)+str(j), crop_img)
                plt.plot(hist_crop)

# pour chaque zone ressemblante, on trace un rectangle
for (pos_x, pos_y) in result:
    cv2.rectangle(img_rgb, (pos_y, pos_x), (pos_y + h, pos_x + w), (0, 255, 255), 2)


img_rgb = cv2.resize(img_rgb, (800, 600))
cv2.imshow('Detected', img_rgb)
cv2.waitKey(0)
cv2.destroyAllWindows()
