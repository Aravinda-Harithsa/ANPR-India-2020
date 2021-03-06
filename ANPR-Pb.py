##########################################################################################################################################
# Author details :- This code was Written by Team consisting of Aravinda Harithsa ,Vijay shekar,Maheshwari Das , Vachana M 
#                                    Digital Image processing Project 
#                            Sri Jayachamarajendra college of engineering
#                          Electronics and communication engineering (JSSSTU) 
#     Automatic Traffic liscense plate detection for Indian cars with deblurring, OCR, and RTO information retreval 
#                               (2020- April -Final code of the project) 
##########################################################################################################################################
# here are list of various libraries imported for various functionalities
import pandas as pd
import numpy as np
import math
import copy
import cv2
import time
import os
import csv
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from keras.models import Sequential, load_model
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten
from matplotlib import pyplot as plt

import argparse
import sys
import os.path
import pytesseract
import cv2
###########################################################################################################################################
'''This function implements Blurring for reduction of noise as a part of preprocessing '''
###########################################################################################################################################

def Smoothen_image(img, d=20):
    h, w  = img.shape[:2]
    img_pad = cv2.copyMakeBorder(img, d, d, d, d, cv2.BORDER_WRAP)
    img_blur = cv2.GaussianBlur(img_pad, (2*d+1, 2*d+1), -1)[d:-d,d:-d]
    y, x = np.indices((h, w))
    dist = np.dstack([x, w-x-1, y, h-y-1]).min(-1)
    w = np.minimum(np.float32(dist)/d, 1.0)
    return img*w + img_blur*(1-w)


###########################################################################################################################################
'''This sub module of the program will rectify the motion blurring caused with required PSF values '''
###########################################################################################################################################
def motion_rectifier(angle, d, sz=120):
    kernel = np.ones((1, d), np.float32)
    c, s = np.cos(angle), np.sin(angle)
    A = np.float32([[c, -s, 0], [s, c, 0]])
    sz2 = sz // 2
    A[:,2] = (sz2, sz2) - np.dot(A[:,:2], ((d-1)*0.7, 0))
    kernel = cv2.warpAffine(kernel, A, (sz, sz), flags=cv2.INTER_CUBIC)
    return kernel

###########################################################################################################################################
'''This function is utilized to crop out the detected number plate from the whole image for further processing'''
###########################################################################################################################################
def cropped(left, top, leftwidth, topheight):
    crop_img = frame[top:topheight, left:leftwidth]
    
    cv2.imwrite("cropped.jpg",crop_img.astype(np.uint8))
    
###########################################################################################################################################
'''This provides the names of the layers which are utilized in the network'''
###########################################################################################################################################
def getOutputsNames(net):
    # Get the names of all the layers in the network
    layersNames = net.getLayerNames()
    # Get the names of the output layers, i.e. the layers with unconnected outputs
    return [layersNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]

###########################################################################################################################################
'''This is to draw the boxes around the predicted objects and articles '''
###########################################################################################################################################
def drawPred(classId, conf, left, top, right, bottom):
    # To Draw a bounding box on the output image 
    cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 3)
    label = '%.2f' % conf
    # To Get the label for the class name and its confidence
    if classes:
        assert(classId < len(classes))
        label = '%s:%s' % (classes[classId], label)

    #Display the label at the top of the bounding box
    labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    top = max(top, labelSize[1])
    cv2.rectangle(frame, (left, top - round(1.5*labelSize[1])), (left + round(1.5*labelSize[0]), top + baseLine), (0, 0, 255), cv2.FILLED)
    #cv.rectangle(frame, (left, top - round(1.5*labelSize[1])), (left + round(1.5*labelSize[0]), top + baseLine),    (255, 255, 255), cv.FILLED)
    cv2.putText(frame, label, (left, top), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,0), 2)


###########################################################################################################################################
'''.This will  Remove the bounding boxes with low confidence using non-maxima suppression
with predefined threshold'''
###########################################################################################################################################

def postprocess(frame, outs):
    print("YOLO v3 based CNN  pre trained model running for detection of LP\n")
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]

    classIds = []
    confidences = []
    boxes = []
    # Scan through all the bounding boxes output from the network and keep only the
    # ones with high confidence scores. Assign the box's class label as the class with the highest score.
    classIds = []
    confidences = []
    boxes = []
    for out in outs:
        print("out.shape : ", out.shape)
        for detection in out:
            #if detection[4]>0.001:
            scores = detection[5:]
            classId = np.argmax(scores)
            #if scores[classId]>confThreshold:
            confidence = scores[classId]
            if detection[4]>confThreshold:
                print(detection[4], " - ", scores[classId], " - th : ", confThreshold)
                print(detection)
            if confidence > confThreshold:
                center_x = int(detection[0] * frameWidth)
                center_y = int(detection[1] * frameHeight)
                width = int(detection[2] * frameWidth)
                height = int(detection[3] * frameHeight)
                left = int(center_x - width / 2)
                top = int(center_y - height / 2)
                classIds.append(classId)
                confidences.append(float(confidence))
                boxes.append([left, top, width, height])

    # Perform non maximum suppression to eliminate redundant overlapping boxes with
    # lower confidences.
    indices = cv2.dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThreshold)
    for i in indices:
        i = i[0]
        box = boxes[i]
        left = box[0]
        top = box[1]
        width = box[2]
        height = box[3]
        cropped(left, top, left + width, top + height)
        drawPred(classIds[i], confidences[i], left, top, left + width, top + height)


###########################################################################################################################################
""" This function resize non square image to square one (height == width)
    """
###########################################################################################################################################

def square(img):
    
    # image after making height equal to width
    squared_image = img

    # Get image height and width
    h = img.shape[0]
    w = img.shape[1]

    # In case height superior than width
    if h > w:
        diff = h-w
        if diff % 2 == 0:
            x1 = np.zeros(shape=(h, diff//2))
            x2 = x1
        else:
            x1 = np.zeros(shape=(h, diff//2))
            x2 = np.zeros(shape=(h, (diff//2)+1))

        squared_image = np.concatenate((x1, img, x2), axis=1)

    # In case height inferior than width
    if h < w:
        diff = w-h
        if diff % 2 == 0:
            x1 = np.zeros(shape=(diff//2, w))
            x2 = x1
        else:
            x1 = np.zeros(shape=(diff//2, w))
            x2 = np.zeros(shape=((diff//2)+1, w))

        squared_image = np.concatenate((x1, img, x2), axis=0)

    return squared_image

###########################################################################################################################################
'''After cropping the plate the image is sent for post processing with thresholding, binirization,Contour sorting and many more to get
end result of sorted countours from left to right'''
###########################################################################################################################################
def plate_segmentation(img_file_path):

    img = cv2.imread(img_file_path)
    gray1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    width = int(img.shape[1]*3)
    height = int(img.shape[0]*3)
    dim = (width, height)
    # resize image
    resized = cv2.resize(gray1, dim, interpolation = cv2.INTER_AREA)
    gray = cv2.threshold(resized, 0, 255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    gray = cv2.medianBlur(gray, 3)
    kernele = np.ones((3,3),np.uint8)
    kerneld = np.ones((5,5),np.uint8)
    #opening = cv2.morphologyEx(imgray, cv2.MORPH_OPEN, kernele)
    erosion = cv2.erode(gray,kernele,iterations = 1)
    dilation = cv2.dilate(erosion,kerneld,iterations = 1)
    cv2.imshow("thebestu", dilation)
    cv2.imwrite("thebestu.jpg", dilation)
    detectu = cv2.imread("thebestu.jpg")
    ret3,th3 = cv2.threshold(dilation,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    
    height = dilation.shape[0]
    width = dilation.shape[1]
    area = height * width

    scale1 = 0.01
    scale2 = 0.2
    area_condition1 = area * scale1
    area_condition2 = area * scale2  
    contours, hierarchy = cv2.findContours(th3, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # sort contours
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    x_cntr_list = []
    cropped = []
    for cnt in contours:
        (x,y,w,h) = cv2.boundingRect(cnt)


        if (w * h > area_condition1 and w * h < area_condition2 and w/h > 0.2 and h/w > 0.3):
            x_cntr_list.append(x) 
            cv2.drawContours(detectu, [cnt], 0, (0, 255, 0), 3)
            cv2.rectangle(detectu, (x,y), (x+w,y+h), (255, 0, 0), 2)
            c = dilation[y:y+h,x:x+w]
            c = np.array(c)
            c = cv2.bitwise_not(c)
            c = square(c)
            
            c = cv2.resize(c,(28,28), interpolation = cv2.INTER_AREA)
            cropped.append(c)
    cv2.imwrite('detection.png', detectu)
    indices = sorted(range(len(x_cntr_list)), key=lambda k: x_cntr_list[k])
    croppedw = []
    for idx in indices:
       croppedw.append(cropped[idx])# stores character images according to their index
    
    
    return croppedw

###########################################################################################################################################
'''Here the main code of the program will start '''
###########################################################################################################################################
#-==----------------===============================================
if __name__ == '__main__':
    print("------------WELCOME TO INDIAN TRAFFIC SURVAILANCE( INDIAN ANPR 2020-------------------------")
    # Initialize the parameters
    confThreshold = 0.5  #Confidence threshold
    nmsThreshold = 0.4  #Non-maximum suppression threshold
    inpWidth = 416  #608     #Width of network's input image
    inpHeight = 416 #608     #Height of network's input image
    parser = argparse.ArgumentParser(description='Object Detection using YOLO in OPENCV')
    

    # Load names of classes
    classesFile = "classes.names";
    classes = None
    with open(classesFile, 'rt') as f:
        classes = f.read().rstrip('\n').split('\n')

    # Give the configuration and weight files for the model and load the network using them.
    modelConfiguration = "darknet-yolov3.cfg";
    modelWeights = "lapi.weights";

    net = cv2.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)    

    # Process inputs
    
    outputFile = "yolo_out_py.avi"
    frame = cv2.imread("test.jpg")
    outputFile = 'yolo_out_py.jpg'
       
        # Create a 4D blob from a frame.
    blob = cv2.dnn.blobFromImage(frame, 1/255, (inpWidth, inpHeight), [0,0,0], 1, crop=False)

        # Sets the input to the network
    net.setInput(blob)

        # Runs the forward pass to get output of the output layers
    outs = net.forward(getOutputsNames(net))

        # Remove the bounding boxes with low confidence
    postprocess(frame, outs)
        
        # Put efficiency information. The function getPerfProfile returns the overall time for inference(t) and the timings for each of the layers(in layersTimes)
    t, _ = net.getPerfProfile()
    label = 'Inference time: %.2f ms' % (t * 1000.0 / cv2.getTickFrequency())
        #cv.putText(frame, label, (0, 15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))

        # Write the frame with the detection boxes
    if ('test.jpg'):
        print("LP image generated and saved\n")
        cv2.imwrite(outputFile, frame.astype(np.uint8));
    else:
        vid_writer.write(frame.astype(np.uint8))
   

        
      
    cv2.imshow("outputFile", frame.astype(np.uint8))
    print("Deblurring Sucessfully DOne .......\n")
    print("----------------------------------------------------------------")
    
    
    print("----------------------------------------------------------------")
    print("Recognition OCR running ....")
    print("----------------------------------------------------------------")
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'   

   
    
    
    # Load model
    model = load_model('cnn_classifier.h5')
    total=0
    count=0
    # Detect chars
    #crp = cv2.imread("cropped.jpg")
    #cv2.namedWindow('Number plate cropped', cv2.WINDOW_NORMAL)
    #cv2.resizeWindow("Number plate cropped", 250,100)
    #cv2.moveWindow("Number plate cropped", 0, 300)
    #cv2.imshow("Number plate cropped",crp)
    
    digits = plate_segmentation('cropped.jpg')
    result =[]
    # Predict
    for d in digits:

        d = np.reshape(d, (1,28,28,1))
        out = model.predict(d)
        # Get max pre arg
        p = []
        precision = 0
        for i in range(len(out)):
            z = np.zeros(36)
            z[np.argmax(out[i])] = 1.
            precision = max(out[i])
            p.append(z)
        prediction = np.array(p)

        # Inverse one hot encoding
        alphabets = ['0','1','2','3','4','5','6','7','8','9','A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']
        classes = []
        
        for a in alphabets:
            classes.append([a])
        ohe = OneHotEncoder(handle_unknown='ignore', categorical_features=None)
        ohe.fit(classes)
        pred = ohe.inverse_transform(prediction)
        print('Prediction : ' + str(pred[0][0]) + ' , Precision : ' + str(precision))

        if precision > 0.5:
            #print('Prediction : ' + str(pred[0][0]) + ' , Precision : ' + str(precision))
            result.append(str(pred[0][0]))
            total = total+precision
            count=count+1
    img = cv2.imread("thebestu.jpg")
    gray1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
  
    cv2.imshow("ROI detection ", cv2.imread("yolo_out_py.jpg"))
    
    cv2.imshow("Image segmentation",cv2.imread("detection.png"))
    
    print("###################################################################################### ")            
    print("\nvehicle number by CNN model 1 ( Tesseract ocr) ")
    text = pytesseract.image_to_string(cv2.imread("thebestu.jpg"))
    print(text)


           

   
    
#-=============OCR BY SELF TRAINED MODEL==================================================================== ---------------------
    str1 = ""      
    for ele in result:
        str1 += ele
    print("###################################################################################### ") 
    print("\nThe vehicle number by CNN model 2 Tensorflow -MNIST dataset : ")
    print(str1)
    print("###################################################################################### ") 
    accuracy=(total/count)*100
    print("\nThe Precision of detection is : ",int(accuracy),"percent")

    rto=""
    for element in range(0,4): 
        rto+=str1[element]   
    filename = "indian RTO.csv"  
    with open(filename, 'r') as csvfile: 
        
        csvreader = csv.reader(csvfile) 
        fields = next(csvreader) 
        for row in csvreader:
            if row[1]== rto:
                print(row[1],row[2],row[3])
                    
#==========================ocr By tesseract================================================================
    rto1=""
    print(text)
    text.replace(" ", "")
    print(text)
    for element in range(0,6):
        if(text[element]==" "):
            continue 
        rto1+=text[element]
        if(element==4):
            break
    with open(filename, 'r') as csvfile: 
        
        csvreader = csv.reader(csvfile) 
        fields = next(csvreader) 
        for row in csvreader:
            if row[1]== rto1:
                print("###################################################################################### ") 
                print(" The vehicle details are :- \n")
                print(row[1],row[2],row[3])
                print ( "visit vahan.nic.in for more details")
                print("###################################################################################### ")     

#---------------------------------------------------------------------------



            

