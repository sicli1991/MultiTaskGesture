# -*- coding: utf-8 -*-
#Set Up Modules:
#--------------------------------------------------------------------------------------
import numpy as np                 #library for working with arrays
import matplotlib.pyplot as plt    #libary for plotting (extension of numpy)
import re as regex                 #library for regular expressions
import cv2                         #libary to solve computer vision problems
import math                        #math tools
import os
from os import listdir, sys        #to use "listdir" and "mkdir"
from os.path import isfile, join   #to use file tools
import random                      #library for randomization tools
import time                        #library for timing tools
from itertools import compress     #used to allow boolean to choose elements of list
import copy

#Point to Images:
#--------------------------------------------------------------------------------------
#path to images

imageParentPath = r"C:\Users\Desktop\pro\00-Data\00-Raw_Images(ByUser)\User 1\gesture 2\Left"
imageBaseName = "U1G2L"
NumberOfWristCuts = 10

#image search criteria
RegExpression = '[A-Z0-9_]+\.tiff'

# Variables added for 'PrepareImageAndLabelsDeepLearning'
finalImageSize = [100, 100]   #stored as [row, col] format
imagePadFraction = 0.15       #Fraction of original image size used as buffer pixels around the hand  
fingerNotPresentNo = 9999
debugMode = 0                 #1 for printing debug message and showing debug images, 0 to turn them off 

#find images
allFilesInParentPath = [f for f in listdir(imageParentPath) if isfile(join(imageParentPath, f))]
allFilesInParentPath = ' '.join(allFilesInParentPath)
allFilesInParentPath = regex.findall(RegExpression,allFilesInParentPath)
print('Printing files to screen to show which files are going to be used...')
print('Number of files: ' + str(len(allFilesInParentPath)))
print(allFilesInParentPath)

#make new folder location
os.mkdir(imageParentPath + '\\Prepared')

#Functions
#------------------------------------------  
def Euclidean_Distance(points, reference_point):
    """Calculates a distance between an array of points and a singular reference point."""
    dist = None
    if isinstance(points, tuple):
        dist = 0
        for i in range(0, len(reference_point), 1):
            dist += (points[i] - reference_point[i]) ** 2
        dist = dist ** (1 / 2)
    elif len(points) > 1:
        dist = (np.array(points) - reference_point)**2
        dist = np.sum(dist, axis=1)
        dist = np.sqrt(dist)
    else:
        print('ERROR: First input must be a single tuple (x,y) or a list of tuples [(x,y),(x,y),...]')
    return dist


def Center_of_Mass(points):
    """Calculate center of mass of given point coordinates."""
    x = int(sum([x for x,y in points])/len(points))
    y = int(sum([y for x,y in points])/len(points))
    return (x,y)


def Cut_Off_Wrist(WristPoint1, WristPoint2, Arm_Point, Percent_Arm, img):
    '''Obtain Status of Points relative to inequality line.'''
    #generate points for image (grid of values)
    x_size = np.shape(img)[1]
    y_size = np.shape(img)[0]
    Points = []
    for i in range(0,x_size):
        for j in range(0,y_size):
            Points.append([i,j])

    #obtain inequality based on wrist points
    m = (WristPoint2[1] - WristPoint1[1])/(WristPoint2[0] - WristPoint1[0])
    b = (-1*m*WristPoint1[0] + WristPoint1[1])
    
    #change one wristpoint to midpoint
    CutPoint = Center_of_Mass([WristPoint1,WristPoint2])
    
    #move CutPoint towards Arm_Point by Percent_Arm
    temp = [0,0]
    dx = Arm_Point[0] - CutPoint[0]
    dy = Arm_Point[1] - CutPoint[1]
    temp[0] = CutPoint[0] + int(dx * Percent_Arm)
    temp[1] = CutPoint[1] + int(dy * Percent_Arm)
    b_ = (-1*m*temp[0] + temp[1])
    
    #separate x and y values for matrix operations
    x_groundTruth = np.array([x[0] for x in Points])
    y_groundTruth = np.array([x[1] for x in Points])
       
    #calculate value of y_fromInequality
    y_fromInequality = np.array(m*(x_groundTruth) + b_)
    
    #sort points by inequality
    Side_1 = y_groundTruth >= y_fromInequality
    Side_2 = y_groundTruth < y_fromInequality
    
    #report points belonging to side opposite of COP
    index = Points.index(Arm_Point)
    if Side_1[index] == True:
        Points_ToDelete = list(compress(Points, Side_1))
    else:
        Points_ToDelete = list(compress(Points, Side_2))
         
    #cut off wrist
    for x,y in Points_ToDelete:
        img[y,x] = 0
    return img
        

def resizeImage(img, desired_size):
    """Scales an image to a desired size."""
    X_image = np.shape(img)[1]
    Y_image = np.shape(img)[0]
    
    #scale carefully, to keep aspect ratio
    if X_image > Y_image:
        width = int(desired_size[0])
        height = math.ceil((width/X_image)*Y_image)
    else:
        height = int(desired_size[1])
        width = math.ceil((height/Y_image)*X_image)
    dim = (width, height)
    return cv2.resize(img, (desired_size[0], desired_size[1]), interpolation=cv2.INTER_AREA)
    
    
def padImage(image, desired_size):
    """Pads an image with a constant color (black) to a desired size."""
    #find amount of X and Y padding needed
    X_Pad_Needed = max(0,desired_size[0] - np.shape(image)[1])
    Y_Pad_Needed = max(0,desired_size[1] - np.shape(image)[0])
    
    #define even split of padding
    left = int(X_Pad_Needed/2)
    right = int(X_Pad_Needed/2)
    top = int(Y_Pad_Needed/2)
    bottom = int(Y_Pad_Needed/2)        
     
    #if needed padding is odd, use floor and ceil
    if X_Pad_Needed % 2 > 0 and X_Pad_Needed > 0:
        left = int((X_Pad_Needed-1)/2)
        right = int((X_Pad_Needed+1)/2)
    if Y_Pad_Needed % 2 > 0 and Y_Pad_Needed > 0:
        top = int((Y_Pad_Needed-1)/2)
        bottom = int((Y_Pad_Needed+1)/2)
    return cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT)    
    
    
def Prepare_Hand(event, x, y, flags, param):
    '''
        Prepares region of interest to select only the bounding box area. 
        This allows for a multitude of slightly removed regions of interest
        to be refined to be the same hand region on interest before attempting
        to pad and resize for the final time.
    '''
    #declarations
    global bufferPoints, image, start, end
   
    #left-click
    if event == cv2.EVENT_LBUTTONDBLCLK:
        bufferPoints.append([x, y])
        print(bufferPoints)
    elif event == cv2.EVENT_RBUTTONDOWN:
        #top-left corner of image (as displayed)
        start = [x,y]
    elif event == cv2.EVENT_RBUTTONUP:
        #crop image based on user's mouse operations
        #bottom-right corner of image (as displayed)
        end = [x,y]
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        desired_size = max(dx,dy)
        image = image[start[1]:end[1],start[0]:end[0]]
        
        #obtain contours, findbounding box
        contours,hierarchy = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        x_box,y_box,dx,dy = cv2.boundingRect(contours[0])
        
        #crop image based on this one bounding box
        image = image[y_box:(y_box+dy),x_box:(x_box+dx)]
        
        #magnify image for ease of selecting points
        image = resizeImage(image, (200,200)) 
        print(np.shape(image))
        

def PrepareImageAndLabelsDeepLearning(img, w, f):
    '''
        Adds border to image, then resizes to final desired size.
    '''
    row, col = img.shape
    minRow = row
    minCol = col
    maxRow = 0
    maxCol = 0
    for r in range(0,row):
        for c in range(0,col):
            if image[r, c] > 0:
                if minRow > r:
                    minRow = r
                if minCol > c:
                    minCol = c
                if maxRow < r:
                    maxRow = r
                if maxCol < c:
                    maxCol = c    
     
    ImgWidth  = maxCol - minCol + 1
    ImgHeight = maxRow - minRow + 1
    pad = round( imagePadFraction * max(finalImageSize[0], finalImageSize[1]) )
    
    top = pad
    bottom = pad
    left = pad
    right = pad
    if ImgWidth > ImgHeight:
        top = pad + round((ImgWidth - ImgHeight)/2)
        bottom = pad + (ImgWidth - ImgHeight) - round((ImgWidth - ImgHeight)/2)
    else:
        left  = pad + round((ImgHeight - ImgWidth)/2)
        right = pad + (ImgHeight - ImgWidth) - round((ImgHeight - ImgWidth)/2)
    
    imagePadded = cv2.copyMakeBorder(img[minRow:maxRow, minCol:maxCol], top, bottom, left, right, cv2.BORDER_CONSTANT)
    imagePaddedResize = resizeImage(imagePadded, (finalImageSize[1], finalImageSize[0]))
    #imagePadded = cv2.copyMakeBorder(image, pad, pad, pad, pad, cv2.BORDER_CONSTANT)
    #imagePaddedResize = cv2.resize(imagePadded, (finalImageSize[1], finalImageSize[0]), interpolation=cv2.INTER_AREA)
    if debugMode:
        print("top, bottom, left, right, pad, ImgWidth, ImgHeight:", top, bottom, left, right, pad, ImgWidth, ImgHeight)
        print("Shape of imagePadded: ", imagePadded.shape)
        print("Shape of imagePaddedResize: ", imagePaddedResize.shape)
        #print("minRow: ", minRow, " minCol: ", minCol, " maxRow: ", maxRow, " maxCol: ", maxCol) 
    
    
    #Update label cordinates
    wLenth = len(w)
    rowPad, colPad = imagePadded.shape
    for i in range(0, wLenth):
        #w[i][0] = round( (w[i][0] - minRow + top) * (finalImageSize[0]/rowPad) )
        #w[i][1] = round( (w[i][1] - minCol + left) * (finalImageSize[1]/colPad) )
        w[i][1] = round( (w[i][1] - minRow + top) * (finalImageSize[0]/rowPad) )
        w[i][0] = round( (w[i][0] - minCol + left) * (finalImageSize[1]/colPad) )
    
    fLenth = len(f)
    for i in range(0, fLenth):
        if(f[i][0] != fingerNotPresentNo):
            #f[i][0] = round( (f[i][0] - minRow + top) * (finalImageSize[0]/rowPad) )
            #f[i][1] = round( (f[i][1] - minCol + left) * (finalImageSize[1]/colPad) )
            f[i][1] = round( (f[i][1] - minRow + top) * (finalImageSize[0]/rowPad) )
            f[i][0] = round( (f[i][0] - minCol + left) * (finalImageSize[1]/colPad) )    
    
    
    return imagePaddedResize, w, f

#set WristPoints as global
global bufferPoints, image, image2, image3, start, end

#Algorithm to put wrist points into image and store into array
for i in range(0,len(allFilesInParentPath)): 
    #Open image and binarize it
    imageName = allFilesInParentPath[i]
    imagePath = imageParentPath + '\\' + imageName
    image = cv2.imread(imagePath, cv2.IMREAD_GRAYSCALE)
    image = cv2.threshold(image,80,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
    image2 = image.copy()
    
    #display image, do stuff to it
    cv2.namedWindow("image")
    cv2.setMouseCallback("image", Prepare_Hand)
    bufferPoints = [] 
    w = []
    f = []
    while True:
        #display the image and wait for a keypress
        cv2.imshow("image", image)
        key = cv2.waitKey(1) & 0xFF
        
        #options
        if key == ord("r"):        ##reset image and points (try again)
            print("RESET")
            bufferPoints = []
            w = []
            f = []
            image = image2.copy()
        elif key == ord("w"):      ##store bufferPoints into wrist array
            if len(bufferPoints) == 0:
                print("You did not grab any points. Get some first, then store.")
            elif len(bufferPoints) != 3:
                print("You did not grab enough points. Points array is reset, try again.")
                bufferPoints = []                
            else:
                w = bufferPoints
                bufferPoints = []
                print("WristPoints Captured are: ", w)
        elif key == ord("f"):      ##store bufferPoints into fingers array 
            if len(bufferPoints) == 0:
                print("You did not grab any points. Get some first, then store.")
            elif len(bufferPoints) != 5:
                print("You did not grab enough points. Points array is reset, try again.")
                bufferPoints = []
            else:
                f = bufferPoints 
                print("FingerPoints Captured are: ", f)
        elif key == ord("x"):      ##store "no finger" into bufferPoints
            bufferPoints.append((9999,9999))
            print(bufferPoints)
        elif key == ord("b"):      ##clear buffer points
            bufferPoints = []
            print("Reinitialized bufferPoints")
        elif key == ord("c") or key == ord("q") or key == ord("s"):  #break from loop
            image3 = image.copy()         
            break
      
    #close image window
    cv2.destroyAllWindows()         
    
    #what to do now?
    if key == ord("q"): #quit
        break
    if key == ord("s"): #skip image, go to next
        continue
    
    #cut wrist, and store image
    for j in range(0,int(NumberOfWristCuts)):
        if debugMode:
            print("This is image 3")
            cv2.imshow("image", image3)   #<----You can remove this line if you don't want to watch the cutting
            key = cv2.waitKey(0)         #<----You can remove this line if you don't want to watch the cutting
            cv2.destroyAllWindows()      #<----You can remove this line if you don't want to watch the cutting        
        
        #cut wrist
        image = Cut_Off_Wrist(w[0], w[1], w[2], (j/int(NumberOfWristCuts)), image3.copy())
        if debugMode:
            print("This is the image right after my algorithm")
            cv2.imshow("image", image)   #<----You can remove this line if you don't want to watch the cutting
            key = cv2.waitKey(0)         #<----You can remove this line if you don't want to watch the cutting
            cv2.destroyAllWindows()      #<----You can remove this line if you don't want to watch the cutting
        
        w_curr = copy.deepcopy(w)
        f_curr = copy.deepcopy(f)
        image, w_curr, f_curr = PrepareImageAndLabelsDeepLearning(image.copy(), w_curr, f_curr)  
        image = cv2.threshold(image,80,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
        if debugMode:
            print("This is the image right after Soumya's algorithm")
            cv2.imshow("image", image)   #<----You can remove this line if you don't want to watch the cutting
            key = cv2.waitKey(0)         #<----You can remove this line if you don't want to watch the cutting
            cv2.destroyAllWindows()      #<----You can remove this line if you don't want to watch the cutting
        
            for k in range(2):
                cv2.circle(image, (w_curr[k][0], w_curr[k][1]), 8, (150, 0, 0), 2)
            for k in range(5):
                if f_curr[k][0] != 9999:
                    cv2.circle(image, (f_curr[k][0], f_curr[k][1]), 8, (150, 0, 0), 2)
        
        if debugMode:
            cv2.imshow("image", image)   #<----You can remove this line if you don't want to watch the cutting
            key = cv2.waitKey(0)         #<----You can remove this line if you don't want to watch the cutting
            cv2.destroyAllWindows()      #<----You can remove this line if you don't want to watch the cutting
        
        #Print image to file
        imageOriginal = imageName.split('.')[0]
        imageOriginal = imageOriginal.split('_')[1]        
        imageExtension = imageName.split('.')[1]
        new_image_name = imageBaseName + "_" + imageOriginal + "_" + str(j) + "." + imageExtension
        new_image_path = imageParentPath + '\\Prepared\\' + new_image_name
        writeFlag = 0
        while writeFlag == 0:
            writeFlag = cv2.imwrite(new_image_path,image)

        #Print label to file
        new_image_labels = imageBaseName + "_" + imageOriginal + "_" + str(j) + ".txt" 
        new_labels_path = imageParentPath + '\\Prepared\\' + new_image_labels
        fil = open(new_labels_path,"w+") 
        fil.write(str(w_curr) + str(f_curr)) 
        fil.close()
