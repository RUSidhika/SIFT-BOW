''' This code is used to perform Object Recognition over 5 different classes.
Number of Training Images per class = 20
Number of Test Images per class = 2
To perform Object recognition, SIFT - Bag of Words model is used.  '''

from numpy import *
import numpy as np
import cv2
import os

# Path to training images for 5 classes
path={};
classes=['bottle','can','car','cube','mug'];  
for x in range(len(classes)):
    ##..........CHANGE THE PATH TO TRAINING IMAGES HERE...........##
    path[x]='/home/sidhika/Desktop/Python_Code/training/%s/' %(classes[x])
    

    # path to test images
    ##..........CHANGE THE PATH TO TESTING IMAGES HERE...........##
    test_path="/home/sidhika/Desktop/Python_Code/testing/"


##...initializing parameters...##
I1=0            # starting index of training images
I2=19           # end index of training images
no_img=I2-I1    # no. of images 
clusters=20     # no. of clusters in k-means
no_neighbors=7  # no. of neighbours to be considered for KNN testing

# Function to extract sift features from images of given class and concatenate them into single matrix
def sift_descriptor(Path,I1,I2):
    des_train=None
    size=None
    listing = sorted(os.listdir(Path))
    for image in listing[I1:I2]:
          im=cv2.imread(Path+image)
          # finding SIFT key points and descriptors
          sift = cv2.xfeatures2d.SIFT_create()
          (kp_train, des1_train)=sift.detectAndCompute(im,None)
          size1=np.shape(des1_train)
          if des_train is None:
               des_train=des1_train
               size=np.array([size1])
          else:
              des_train=np.concatenate((des_train,des1_train),axis=0)
              size=np.concatenate((size,np.array([size1])),axis=0)
              
    return des_train, size[:,0]


# Function to create histogram representation of images after K-means clustering over all the SIFT features of all classes.
def create_hist(codewords,clusters,size_class,sum_features,classidx):
    idx=np.concatenate((np.array([0]),size_class))
    range1=np.sum(sum_features[0:classidx])
    range2=range1+sum_features[classidx]
    codewords_temp=codewords[range1:range2]
    R1=np.shape(idx)
    hist=None
    for i in range(0,R1[0]-1):
        range1=np.sum(idx[0:i+1])
        range2=range1+idx[i+1]
        hist_temp, bin_edges=np.histogram(codewords_temp[range1:range2],clusters)
        if hist is None:
             hist=hist_temp
        else:   
             hist=np.vstack((hist,hist_temp))
    return hist

# Function to encode the given test image into histogram feature.
def create_testfeature(Path,I3,centers,clusters):
    listing = sorted(os.listdir(Path))
    test_img=cv2.imread(Path+listing[I3])
    sift = cv2.xfeatures2d.SIFT_create()
    (kp_train, dest1_train)=sift.detectAndCompute(test_img,None)
    no_descr=np.shape(dest1_train)
    dist = np.zeros((no_descr[0],clusters))
    Min=np.zeros(no_descr[0])
    for i in range(0,no_descr[0]):
        for j in range(0,clusters):      
            a=dest1_train[i,:]
            b=centers[j,:]
            dist[i,j]=np.linalg.norm(a-b)
        Min[i]=np.argmin(dist[i,:])    
    Min1=Min.astype(np.float32)
    hist_test, bin_edges=np.histogram(Min1,clusters) 
    return hist_test.astype(float32)  


def main():
    descr_class={};
    size_class={};
    addi={};

# Extract SIFT Features of images from all class and concatenate into same Matrix
    for x in range(len(classes)):
        descr_class[x], size_class[x]=sift_descriptor(path[x],I1,I2)
        addi[x]=np.sum(size_class[x])

    descriptors=np.concatenate((descr_class[0],descr_class[1],descr_class[2],descr_class[3],descr_class[4]),axis=0)
    sum_features=np.hstack(([0],addi[0],addi[1],addi[2],addi[3],addi[4]))


    #... performing k-means clustering on the descriptors ...#

    # Define criteria = ( type, max_iter = 10 , epsilon = 1.0 )
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    # Set flags (Just to avoid line break in the code)
    flags = cv2.KMEANS_RANDOM_CENTERS
    compactness,codewords,centers = cv2.kmeans(descriptors,clusters,None,criteria,10,flags)
    

    #... create histogram of codewords for every class
    Hist_class={}
    for x in range(len(classes)):
        Hist_class[x]=create_hist(codewords,clusters,size_class[x],sum_features,x+1)

    #... concatenate histograms of training images to feed into the classifier
    trainData=np.concatenate((Hist_class[0],Hist_class[1],Hist_class[2],Hist_class[3],Hist_class[4]),axis=0)
    traindata=trainData.astype(np.float32)

    #... creating labels vector for training images
    k = np.arange(5)
    train_labels1 = np.repeat(k,no_img)[:,np.newaxis]
    train_labels=train_labels1.astype(np.float32)

    #....... training KNN ......#
    knn = cv2.ml.KNearest_create()
    knn.train(traindata,cv2.ml.ROW_SAMPLE,train_labels)

    #... testing KNN ....#
    test_list=sorted(os.listdir(test_path))
    print(test_list) 
    Sz=np.shape(test_list)
    hist_test=None
    for i in range(0,Sz[0]):
        Hist_test1=create_testfeature(test_path,i,centers,clusters)
        if hist_test is None:
           hist_test=Hist_test1
        else:
           hist_test=np.vstack((hist_test,Hist_test1))
    ret, results, neighbours, dist = knn.findNearest(hist_test,no_neighbors)
    groundtruth=np.array([[0],[1],[3],[4],[4],[1],[2],[3],[0],[2]])
    count1=0
    for x in range(len(groundtruth)):
        if results[x]==groundtruth[x]:
              count1=count1+1

    
    # Print the Results
    print("predicted labels:")
    print(results[0])
    print("groundtruth:")
    print(groundtruth[0])
    print("Accuracy:")
    print(count1*100/len(groundtruth))



main()






