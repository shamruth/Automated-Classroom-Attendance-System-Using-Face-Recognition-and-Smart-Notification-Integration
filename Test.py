import cv2
import os
import face_recognition
import pickle
#STEP 1:Gathering image and converting THE OBTAINED BGR@RGB
imagepath='..\Image'
imagepathlist=os.listdir(imagepath)
imagelist=[]
stdid=[]
for image in imagepathlist:
    imagelist.append(face_recognition.load_image_file(os.path.join(imagepath, image)))
    #print(image.split('.')[0])
    stdid.append(image.split('.')[0])
#print(len(imagelist))
#print(stdid)


def encodegen(imglist):
    encodedvalues=[]
    for img in imglist:
        img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encoding = face_recognition.face_encodings(img)[0]
        encodedvalues.append(encoding)
    return encodedvalues

print("ENCODING STARTED")
encodedlst=encodegen(imagelist)
enclstwitid=[encodedlst,stdid]
print("ENCODING ENDED")

#SAVING THE FILE
file=open("EncodedFile.p",'wb')
pickle.dump(enclstwitid, file)
file.close()
print("FILE SAVED")


#LOADING THE FILE
print("LOADING THE ENCODED PICKLE FILE")
file=open("EncodedFile.p",'rb')
enclstwitid=pickle.load(file)
file.close()
Lencodedlst,Lstdid=enclstwitid
#print(Lstdid)
print("ENCODED FILE LOADED")

#TESTING
imgTest=face_recognition.load_image_file("../TEST.jpg")
imgTest=cv2.cvtColor(imgTest,cv2.COLOR_BGR2RGB)

#STEP 2:Identifying the Location of the face In the provided picture
faceloc=face_recognition.face_locations(img)[0] #print(faceloc)returns array of location numbers
encodeimg=face_recognition.face_encodings(img)[0]
cv2.rectangle(img,(faceloc[3],faceloc[0]),(faceloc[1],faceloc[2]),(255,0,0),2)

facelocTest=face_recognition.face_locations(imgTest)[0] #print(faceloc)returns array of location numbers
encodeimgTest=face_recognition.face_encodings(imgTest)[0]
cv2.rectangle(imgTest,(facelocTest[3],facelocTest[0]),(facelocTest[1],facelocTest[2]),(255,0,0),2)

#STEP 3:Comparing the both images and identifying the required image
faceDis=face_recognition.face_distance([encodeimg],encodeimgTest)
result=face_recognition.compare_faces([encodeimg],encodeimgTest)
print(faceDis)
print(result)
cv2.putText(imgTest,f'{result} {round(faceDis[0],2)}',(50,50),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,0,0),2)


#cv2.imshow("imgme",img)
cv2.imshow("imgTest",imgTest)
cv2.waitKey(0)