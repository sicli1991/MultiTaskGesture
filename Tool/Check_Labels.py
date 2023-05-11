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
import re

#--------------------------------------------------------------------------------------
#path to images
ParentPath = r"\Desktop\pro\00-Data\Prepared\CombinedTextFiles"

#image search criteria
RegExpression = '[A-Z0-9_]+\.txt'

#file with results
results = open(ParentPath + '/Results.txt', "a+")

#find images
allFilesInParentPath = [f for f in listdir(ParentPath) if isfile(join(ParentPath, f))]
allFilesInParentPath = ' '.join(allFilesInParentPath)
allFilesInParentPath = regex.findall(RegExpression,allFilesInParentPath)
print('Printing files to screen to show which files are going to be used...')
print('Number of files: ' + str(len(allFilesInParentPath)))
print(allFilesInParentPath)
Total = len(allFilesInParentPath)
Count = 0
for file in allFilesInParentPath:
    Count = Count + 1
    f = open(ParentPath + '/' + file, "r")
    c = f.read()
    if len(re.findall(",", c)) != 14:
        STRING = str(Count) + ":" + str(Total) + ":" + c + "<---" + file + " [error]\n"
        print(STRING)
        results.write(STRING)
    else:
        STRING = str(Count) + ":" + str(Total) + "-- Good --\n"
        print(STRING)
    f.close()
results.close()