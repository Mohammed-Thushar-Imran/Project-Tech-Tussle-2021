1. Attendance management system using facial recognition

The prime objective of this project is to develop such an attendance system where the attendants attendance will be marked automatically when they turn their camera on during a live meeting. Their identity will be stored in an excel sheet in an arranged manner without any manual intervention.

2. Import the libraies

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



3. Importing images and corresponding information

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
