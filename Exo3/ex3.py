import cv2
import matplotlib.pyplot as plt

# Mettez les images suivantes dans le dossier courant :
# waves.jpg, beach.jpg, bear.jpg, dog.jpg, lake.jpg, moose.jpg, polar.jpg

names = ["waves", "beach", "bear", "dog", "lake", "moose", "polar"]

OPENCV_METHODS = (
    ("Correlation", cv2.HISTCMP_CORREL),
    ("Chi-Squared", cv2.HISTCMP_CHISQR),
    ("Intersection", cv2.HISTCMP_INTERSECT),
    ("Hellinger", cv2.HISTCMP_BHATTACHARYYA))

index = {}
images = {}

for img_name in names:
    img = cv2.imread(img_name + ".jpg")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, None, fx=0.5, fy=0.5)
    images[img_name] = img

    hist = cv2.calcHist(img, [0], None, [256], [0, 256])

    # on norme l'histogramme "in place"
    hist = cv2.normalize(hist, hist).flatten()
    plt.plot(hist)
    plt.title(img_name)
    index[img_name] = hist

result = {}
for (k, hist) in index.items():
    # on choisit de comparer l'image waves.jpg aux autres images
    comp = cv2.compareHist(index["waves"], hist, cv2.HISTCMP_BHATTACHARYYA)
    result[k] = comp
result = sorted([(v, k) for (k, v) in result.items()])

fig = plt.figure("Query")
ax = fig.add_subplot(1, 1, 1)
ax.imshow(images["waves"])
plt.axis("off")

fig = plt.figure("Results")
#  plus on est proche de 0, plus les histogrammes des images sont proches (la comparaison de l'image avec elle-même donne donc 0)
for (i, (v, k)) in enumerate(result):
    ax = fig.add_subplot(1, len(images), i + 1)
    ax.set_title("%s: %.2f" % (k, v))
    plt.imshow(images[k])
    plt.axis("off")

plt.show()
