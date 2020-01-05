import numpy as np
import cv2
from matplotlib import pyplot as plt

#fonction qui utilise la caméra de l'ordinateur pour prendre des photos de visages
#name = nom de la personne, aussi utilisé comme nom de dossier d'enregistrement
#mirror = image retournée (si la webcam ne le fait pas automatiquement)
#path = chemin du dossier d'enregistrement
#/!\ tous les dossiers doivent déjà être créés
def cam(name, mirror=False, path=''):
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    cap = cv2.VideoCapture(0) # on lance la capture video
    current_frame = 0
    count = 0
    while (cap.isOpened):#tant que la caméra fonctionne
        ret, frame = cap.read()
        current_frame+=1
        if ret == True :
            if mirror == True:#on retourne l'image si elle a étée retournée (effet miroir)
                frame = cv2.flip(frame, 1)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)#on détecte les visages sur l'image
            for (x,y,w,h) in faces:
                roi_gray = gray[y:y+h, x:x+w]
                crop_img = frame[y:y+h, x:x+w]
                if current_frame%24 == 0: #on enregister une image rognée chaque seconde environ (pour une caméra 24 ips)
                    cv2.imwrite(path+name+'/'+name+'_'+str(count)+'.jpg', crop_img)
                    count+=1
                cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)#on ajoute un rectangle autours des visages détectés sur l'image prise par la caméra
            cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): #quitte l'application quand on appuie sur 'q'
            break

cam(name = 'etienne', mirror = True, path = 'dataset/')

cv2.waitKey(0)
cv2.destroyAllWindows()
