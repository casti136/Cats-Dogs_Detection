import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import random
import csv

DIR = '/Users/yvielcastillejos/Downloads/kagglecatsanddogs_3367a/PetImages'
#print(os.listdir(DIR))
CATEGORIES = ["Dog", "Cat"]
data_temp = []
train_data= []
features=[]
label=[]
X= []
Y =[]

def datatrain():
    for category in CATEGORIES:
       path = os.path.join(DIR, category) #Gets Us the path to dog or cat folder
       cnum = CATEGORIES.index(category) # 0 for dogs and 1 for cats
       pixel = 50
       for img in os.listdir(path): #for each element in the listdir path (folder)
         try:
            img_matrix = cv2.imread(os.path.join(path,img), cv2.IMREAD_GRAYSCALE) # 1. gets path to each image, 2. can convert each image to gray, will have RGB (3x3)
            new_img_matrix = cv2.resize(img_matrix, (pixel,pixel))
            data_temp.append( [new_img_matrix,cnum])
            #print(data_train)
            #plt.imshow(new_img_matrix, cmap = 'gray')
            #plt.show()
         except:
            pass
    # Shuffle Data Set
    random.shuffle(data_temp)
    return data_temp

def separate_data(training_data):
    for i in range(0,len(training_data),1):
        features.append(training_data[i][0])
        label.append(training_data[i][1])
    return features,label

#Get DATA from PICTURES
train_data = datatrain()
#Separate into two lists
X,Y = separate_data(train_data) #X is features (image), Y is label (0 or 1)

#SAVE the data points
with open('DataX.csv', "w") as f:
  thewriter = csv.writer(f)
  for i in range(0,len(X),1):
    thewriter.writerow([X[i]])
with open('DataY.csv', "w") as f:
  thewriter = csv.writer(f)
  for i in range(0,len(Y),1):
    thewriter.writerow([Y[i]])
