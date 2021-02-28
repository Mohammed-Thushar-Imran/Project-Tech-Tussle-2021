1. Attendance management system using highly sophisticated facial recognition

The prime objective of this project is to develop such an attendance system where the attendant's attendance will be marked automatically when they turn their camera on during a live meeting. Their identity will be stored in an excel sheet in an arranged manner without any manual intervention.

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

We employ the os library which allows us to import all the images from our desired database folder 'Spring-2021(CSE161)' at once. Moreover, cv2.imread() function is employed to read the image file. Later, we append the stored images from 'imgFrame' to the list 'images' and subsequently, we append the first element of each of the file name into a different list, namely 'classNames', right after splitting the text in order to terminate the file format (.jpeg) from appearing in the live capturing screen. Afterwards, we print the classnames to verify whether or not the file format has been deducted from the file name.


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
                
7. Find the matches

At this moment, we compare the encodings of the stored images in the database to the current frame (live screen) images to find the best match by employing liner SVM classifier and represent the result by a new function called 'face_distance' which determines the similarities in terms of numbers; the lower the distance, the better the match. Besides, we create a loop that grabs the encoding from 'encodeRealTime' into 'encodeFace' and location from 'faceLocRealTime' into 'faceLoc' and store them in the same loop.

          Code:  for encodeFace, faceLoc in zip(encodeRealTime,faceLocRealTime):  
                 matches = face_recognition.compare_faces(encodeListSavedImages,encodeFace)
                 faceDistance= face_recognition.face_distance(encodeListSavedImages,encodeFace)
                 print(faceDistance)
         
                 matchIndex = np.argmin(faceDistance) #losest distance represent the best match
                 
8. Enclose and Label the identity within each face

By this time, the system computed the lowest possible value for 'faceDistance' and stored the data in 'matchIndex'. By manipulating this data, we can label the known faces with their Names and Ids (faceDistance[matchIndex] < 0.50) and unknown faces with the text 'Unknown' at the same time. We also enclose the faces with separate rectangles using 4 different measurements of 'faceLocRealTime' as the rectangles coordinate and print the known and unknown attendants identity right below each rectangle, but prior to that, we rescale the images into their original form.

         Code:  if faceDistance[matchIndex] < 0.60:
                name = classNames[matchIndex].upper()
                markAttendance(name)
                else:name = 'Unknown'
                y1, x2, y2, x1 = faceLoc  # specifying the 4 different face location measurements within 4 variables
                y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4 
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
                cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
               
9. Mark the attendance in an excel sheet

We generate the automated attendance code. We define a function that takes the image file name as its input argument at the beginning. Then we open an Attendance file which is in csv format. we import the date in (%Year-%Month-%day) format and insert the headings ("Identity","Status","Time") for three different attendance columns. Later we open the Attendance file again and read all the lines, and iterate through each line using a for loop afterwards. Next we can split using comma ‘,’. Lastly, we upload the sequence of name, status and enrollment time for every selected attendant in a differant line to the attendance list.  If the user in the camera already has an entry in the file, the system will not re-enter the user's information in the same file. On the other hand if the user is new, then the name of the user along with the enrollment time stamp and status will be stored. We can use the datetime class in the date time package to get the current time.


          Code: def markAttendance(name):

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
                
10. View the webcam/live screen

In order to observe the webcam frontal view/ live capturing screen in our device, we initiate 'cv2.imshow()' function.

          Code: cv2.imshow('Webcam',img)
                cv2.waitKey(1)
                
11. Turn off the live screen/webcam instantaneously

If anything unpleasant occurs in an attendant webcam, the controller can immediately turn the live capturing screen off just by clicking on the 'q' button for 10ms using cv2.waitkey() function.

          Code: if cv2.waitKey(10) & 0xFF == ord ('q'):
                break


Bugs and Future work:

One of the issues that arised while implementing the project is that if we try to cover the whole monitor screen just by incresing the aspect ratio, it will require a device with high computational power or it may slow down the system's execution. Moreover, when the image is fed to a pretrained neural network, it generates 128 measurements that are unique to that particular face but all these measurements can not be known as these are what the model learns by itself. Moreover, the system had to perform an additional subsequent scale up and scale down operation concerning the size of the images in order to boost the speed. 

To mention a far-reaching plan about this project is to develop an online meeting app equipped with facial_recognition or fingerprint based biometric method as a dedicated specification to mark the attendance of an online meeting which will be applicable for both educational institution's online class and official meeting. It has to be user friendly and easily accessible.

           


                 
                
            
        
