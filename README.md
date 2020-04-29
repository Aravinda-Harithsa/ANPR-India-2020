# ANPR-India-2020
 Traffic Survailance system (Liscence plate identification) using 2 different  Models of CNN 

Download the required weights here and paste it into main working directord of this project 

Here is the link : 
https://www.kaggle.com/achrafkhazri/yolo-weights-for-licence-plate-detector



This contains two different variants of code one with deblurring and another with plain images ,download and run both 

Step 1 : Paste the input file named as input.jpg into the directory ( project folder)

step -2 : use cmd, terminal or any IDE to run the code and to see the output


Important Features of this project 

1.Deblurring the image using the decolvolution method ( Weiner Filter).

2.ROI detection using yolo v3 (CNN model) with pretrained  weights.

3.Image enhancment of the ROI image for further processing.

4.Character by character segmentation of images using countours

5.Then it will be resized into 28 x 28 size and sent it to a trained nural network which was trained using 40,000 Fonts of all charachters from A to Z ,0 to 9 . 

6.Tesseract OCR  Engine is also trained for double verification of the data.

7.Then the Given Registration Number is read from RTO database to give the details about location of the vehicle.

* Future scope is to fetch the details of the owner avaialable at vahan.nic.in

