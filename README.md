1. Attendance management system using facial recognition

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

We employ the os library which allows us to import all the images from our desired database folder 'Spring-2021/CSE161' at once. Moreover, cv2.imread() function is employed to read the image file. Later, we append the images from 'imgFrame' to the list 'images' and subsequently, we append the first element of each of the file name into a different list, namely 'classNames', right after splitting the text in order to terminate the file format (.jpeg) from appearing in the live capturing screen. Afterwards, we print the classnames to verify whether the file format has been deducted from the file name or not


           Code: path= 'imageBasic'
                 images= []
                 classNames= []
                 List = os.listdir(path)
                 #print(List)
                 
                 for cl in List:

                 ImgFrame= cv2.imread(f'{path}/{cl}')
                 images.append(ImgFrame)
                 classNames.append(os.path.splitext(cl)[0])

                 print(classNames) 
                 
4. Determine the encodings of the saved images

We define a new function called 'faceEncodings' in order to find the encodings of all stored images form the list 'images' and append all these encodings in a new list, namely 'encodeList'. When the image file is read with the openCV function imread(), the order of the color of the images is BGR which needs to be converted into RGB with appropriate command. In the end, we call the 'faceEncodings' fuction with all the stored images as its input arguments.

           Code: def faceEncodings(images):
                 encodeList= []
                 for img in images:
                 img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
                 encode = face_recognition.face_encodings(img)[0]
                 encodeList.append(encode)
                 return encodeList
                 
                 encodeListSavedImages = faceEncodings(images)
                 print('Encoding operation of the saved images is completed')
          
