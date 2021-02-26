1. Attendance management system using highly sophisticated facial recognition

The prime objective of this project is to develop such an attendance system where the attendants attendance will be marked automatically when they turn their camera on during a live meeting. Their identity will be stored in an excel sheet in an arranged manner without any manual intervention.

2. Import the libraries

Many of the operations are being performed internally within the system by calling some toolkits from the library which are dedicated to perform specific tasks.

i. Cmake: It generates a native build environment that will compile source code, create libraries and build executables in arbitrary combination.

ii. dlib: a toolkit for making real world machine learning and data analysis applications in C++ used for face detection and facial landmark estimation.

iii. face_recognition: It enables to find and manipulate facial features by locating and outlining each person's eyes, nose, mouth and chin.

iv. Numpy: a python library designed to work with array.

v. python_OpenCV: A dedicated libray of python bindings to solve computer vision problem.

vi. os: This module acts as a bridge. It allows many fuctions to interact with the file system.

           Code: import cv2
                 import numpy as np
                 import face_recognition
                 import os
                 from PIL import ImageGrab
                 from datetime import datetime



3. Import images and corresponding information

We employ the os library which allows us to import all the images from our desired database folder 'Spring-2021/CSE161' at once. Moreover, cv2.imread() function is employed to read the image file. Later, we append the stored images from 'imgFrame' to the list 'images' and subsequently, we append the first element of each of the file name into a different list, namely 'classNames', right after splitting the text in order to terminate the file format (.jpeg) from appearing in the live capturing screen. Afterwards, we print the classnames to verify whether or not the file format has been deducted from the file name.


           Code: path= 'Spring-2021/CSE161'
                 images= []
                 classNames= []
                 List = os.listdir(path)
                 print(List)
                 
                 for cl in List:

                 ImgFrame= cv2.imread(f'{path}/{cl}')
                 images.append(ImgFrame)
                 classNames.append(os.path.splitext(cl)[0])

                 print(classNames) 
                 
4. Determine the encodings of the saved images

We define a new function called 'faceEncodings' in order to find the encodings of all stored images from the list 'images' and append all these encodings in a new list, namely 'encodeList'. When the image file is read with the openCV function imread(), the order of the color of the images is BGR which needs to be converted into RGB with appropriate command. In the end, we call the 'faceEncodings' fuction with all the stored images as its input arguments.

           Code: def faceEncodings(images):
                 encodeList= []
                 for img in images:
                 img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
                 encode = face_recognition.face_encodings(img)[0]
                 encodeList.append(encode)
                 return encodeList
                 
                 encodeListSavedImages = faceEncodings(images)
                 print('Encoding operation of the saved images is completed')
                 
5. Initialize live capturing screen or webcam

We can either initialize the capturing screen mode or webcam frontal view based on our preferance. We can initialize both by employing 2 webcams in our device as well. For this particular project, capturing screen mode is the suitable one. To grab frames from webcam frontal view, we develop a video capturing object in the beginning, and import ImageGrab from PIL library for capturing live screen. 'bbox' specifies specific region (bbox= top,left,width,height). After that, we create a while loop to run the webcam/live screen. Then we read the image from live screen through 'captureScreen' function and subsequently, cap.read() permits to read webcam frontal view image. To increase the computing speed of the device, it's wise to resize the image into 1/4 th of its original size even though we retrieve the original size while displaying. later, we convert it to RGB.

        Code for live screen: def captureScreen(bbox=(300,150,900+150,800+300)): 
                              capScreen = np.array(ImageGrab.grab(bbox))
                              capScreen = cv2.cvtColor(capScreen, cv2.COLOR_RGB2BGR)
                              return capScreen
                              
                              while True:
                              img = captureScreen()
                              imgS = cv2.resize(img,(0,0),None,0.25,0.25)
                              imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB) 
                              
        Code for webcam frontal view: cap = cv2.VideoCapture(0)
                                      while True:
                                      success, img = cap.read()
                                      imgS = cv2.resize(img,(0,0),None,0.25,0.25)
                                      imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
                                      
6. Specify the target face and encode it

Once we initialize the webcam/live screen, locating the face from the real time image (with 4 different measurements: top, bottom, left and right)
becomes our priority that paves the way to identify known and unknown faces and to encode them accurately.

          Code: faceLocRealTime = face_recognition.face_locations(imgS)
                encodeRealTime = face_recognition.face_encodings(imgS,faceLocRealTime) 
                
7. Find the mathes

At this moment, we compare the encodings of the stored images in the database to the current frame (live screen) images to find the best match by employing liner SVM classifier and represent the result by a new function called 'face_distance' which determines the similarities in terms of numbers; the lower the distance, the better the match. Besides, we create a loop that grabs the encoding from 'encodeRealTime' into 'encodeFace' and location from 'faceLocRealTime' into 'faceLoc' and store them in the same loop.

          Code:  for encodeFace, faceLoc in zip(encodeRealTime,faceLocRealTime):  
                 matches = face_recognition.compare_faces(encodeListSavedImages,encodeFace)
                 faceDistance= face_recognition.face_distance(encodeListSavedImages,encodeFace)
                 print(faceDistance)
         
                 matchIndex = np.argmin(faceDistance) #losest distance represent the best match
                 
8. Enclose and Label the identity within each face

By this time, the system computed the lowest possible value for 'faceDistance' and stored the data in 'matchIndex'. By manipulating this data, we can label the known faces with their Names and Ids (faceDistance[matchIndex] < 0.60) and unknown faces with the text 'Unknown' at the same time. We also enclose the faces with separate rectangles using 4 different measurements of 'faceLocRealTime' as the rectangles coordinate and print the known and unknown attendants identity right below each rectangle, but prior to that, we rescale the images into their original form.

         Code:  if faceDistance[matchIndex] < 0.60:
                name = classNames[matchIndex].upper()
                markAttendance(name)
                else:name = 'Unknown'
                y1, x2, y2, x1 = faceLoc  # specifying the 4 different face location measurements within 4 variables
                y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4 
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
                cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

           


                 
                
            
        
