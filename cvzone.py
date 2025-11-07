import cv2
import os
import face_recognition
import pickle
import numpy as np
from datetime import datetime
cap=cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

#ADDING IMAGES INTO A LIST
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

#SCALING DOWN THE IMAGE TO REDUCE CMPT PWR

#ATTENDENCE MARKING

# ATTENDANCE STORAGE
attendance = {}  # {name: last_marked_time}
"""
def mark_attendance(name):
    now = datetime.now()
    if name not in attendance:
        attendance[name] = now
        with open("Attendance.csv", "a") as f:
            f.write(f"{name},{now.strftime('%Y-%m-%d %H:%M:%S')}\n")
        print(f"Attendance marked for {name}")
    else:
        elapsed = (now - attendance[name]).total_seconds()
        if elapsed > 60:  # 90 seconds gap
            attendance[name] = now
            with open("Attendance.csv", "a") as f:
                f.write(f"{name},{now.strftime('%Y-%m-%d %H:%M:%S')}\n")
            print(f"Attendance updated for {name}")
"""
#while(True):
   # sucess,img=cap.read()
    img=face_recognition.load_image_file("../TEST.jpg")
    img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #imgS=cv2.resize(img,(0,0),None,0.25,0.25)
    #imgS=cv2.cvtColor(imgS,cv2.COLOR_BGR2RGB)

    facesincurframe=face_recognition.face_locations(img)
    encodecurrframe=face_recognition.face_encodings(img,facesincurframe)


    for encodeface,facelocation in zip(encodecurrframe,facesincurframe):
        matches=face_recognition.compare_faces(Lencodedlst,encodeface)
        facedistance=face_recognition.face_distance(Lencodedlst,encodeface)
        print("matches",matches)
        print("facedistance",facedistance)
        matchindex=np.argmin(facedistance)
        print("matchindex",matchindex)

        if matches[matchindex]:
            print(stdid[matchindex])
    #        mark_attendance(stdid[matchindex])
            y1, x2, y2, x1 = facelocation
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, stdid[matchindex], (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
    cv2.imshow("imgTest", img)
    #cv2.imshow("webcam",img)
    #if cv2.waitKey(1)& 0xFF == ord('q'):
     #   break
#cap.release()
#cv2.destroyAllWindows()
#"""