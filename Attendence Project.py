import cv2
import numpy as np
import face_recognition
import os
from PIL import ImageGrab
from datetime import datetime

#Loacating the folder that contains the images
path= 'Spring-2021 (CSE161)'
images= []
classNames= []
List = os.listdir(path)
#print(List)

for cl in List:

    ImgFrame= cv2.imread(f'{path}/{cl}')
    images.append(ImgFrame)
    classNames.append(os.path.splitext(cl)[0])

print(classNames)
def faceEncodings(images):
    encodeList= []
    for img in images:
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
        #append encoding into the list
    return encodeList

def markAttendance(name):

    with open('Attendance.csv','r+') as s:
        tday = datetime.today().strftime('%Y-%m-%d')
        s.write(f'Date:{tday}')
        s.write(f'\n{"Identity"},{"Status"},{"Time"}')
        s.close()
    with open('Attendance.csv','r+') as f:
        myDataList = f.readlines()

        nameList = []

        for line in myDataList:
          entry = line.split(',')
          nameList.append(entry[0])


        if name not in nameList:
          now = datetime.now()
          dtString = now.strftime('%H:%M:%S')
          f.writelines(f'\n{name},{"Present"},{dtString}')

## FOR CAPTURING SCREEN RATHER THAN WEBCAM
def captureScreen(bbox=(400,200,900+200,800+400)): ##bbox specifies specific region (bbox= x,y,width,height *starts top-left)
      capScreen = np.array(ImageGrab.grab(bbox))
      capScreen = cv2.cvtColor(capScreen, cv2.COLOR_RGB2BGR)
      return capScreen


encodeListSavedImages = faceEncodings(images) #run
print('Encoding operation of the saved images is completed ')

#For capturing using webcam
#initialize the webcam

#cap = cv2.VideoCapture(0)

while True:
    #success, img = cap.read() #capture in real time
    img = captureScreen()
    imgS = cv2.resize(img,(0,0),None,0.25,0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    faceLocRealTime = face_recognition.face_locations(imgS) #loacate the face from the real time image with 4 different measurements from top, bottom, left and right


    encodeRealTime = face_recognition.face_encodings(imgS,faceLocRealTime) #encode the pointed face
    #encode the real time images

    for encodeFace, faceLoc in zip(encodeRealTime,faceLocRealTime):
         #iterate through all the faces that we have found in current frame, then compare them with the encoding of stored faces
         # Create a loop that grab the encoding from encodeRealTime into encodeFace and location from faceLocRealTime into faceLoc and store them in the same loop
        matches = face_recognition.compare_faces(encodeListSavedImages,encodeFace)
        faceDistance= face_recognition.face_distance(encodeListSavedImages,encodeFace)
         #lowest distance would be the desired match
         #sending a list of the encodings of known image as encodeListKnown, it will be returning a list
        print(faceDistance)
         #losest distance represent the best match
        matchIndex = np.argmin(faceDistance)
         #define the best match with '0' and 1 otherwise

        if faceDistance[matchIndex] < 0.50:
             name = classNames[matchIndex].upper()
             markAttendance(name)
         # identify & print the name of the person (import from classNames) in real time image that best match with known image
        else:name = 'Unknown'

        y1, x2, y2, x1 = faceLoc
        y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
         # specifying the 4 different face location measurements within 4 variables
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
         # x1,y1,x2,y2 are the coordinates of the rectangle
        cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
        cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
         # name will be displayed within the image in a rectangle shaped box filled in green color as commanded earlier

    cv2.imshow('Live Screen',img)
    cv2.waitKey(1)
    # to view the real time image in actual size

    # In case of any inconvenience, instantaneously, stop taking video by a single press on 'q'.

    # To stop taking video, press 'q' for at least 10ms
    if cv2.waitKey(10) & 0xFF == ord ('q'):
            break





