from __future__ import division, print_function, absolute_import
import os
import sys
import cv2
import time
import random
import pydicom
import sagemaker
import numpy as np
import tensorflow as tf
sys.path.append('support_funcs')
from shutil import rmtree
from statistics import mean
from sagemaker import get_execution_role
from sagemaker.tensorflow import TensorFlowModel
from tensorflow.python.keras.preprocessing.image import load_img
from echoanalysis_tools import output_imgdict, output_imgdict_still, read_dicom, read_dicom_still, extract_imgs_from_dicom

HEIGHT = 224
WIDHT = 224
QUALITY = 95

def classify(directory, predictor):
    """
    Classifies echo images in given directory
    
    @param directory: folder with jpg echo images for classification
    """

    imageDict = {}
    predictions = {}
    for filename in os.listdir(directory):
        if "jpg" in filename:
            image = load_img(directory + filename, target_size=(HEIGHT, WIDTH))
            imageDict[filename] = np.array(image).reshape((1, HEIGHT, WIDTH, 3))
        
    for filename in imageDict:
        result = predictor.predict({'inputs_input': imageDict[filename]})
        predictions[filename] = result["predictions"][0]

    return predictions


def main():
    total_x = time.time()
    total_dcm = 0
    
    role = get_execution_role()
    bucket = "sagemaker-ap-southeast-2-611188727347"
    key = "tensorflow-training-2020-11-09-01-37-28-348"
    modelPath = "s3://{}/{}/output/model.tar.gz".format(bucket, key)

    # Load model
    model = TensorFlowModel(model_data=modelPath, role=role, framework_version='2.1.0')
    # Deploy model
    predictor = model.deploy(initial_instance_count=1, instance_type='ml.m5.xlarge')
    
    infile = open("model_training/class_labels.txt")  #class labels
    infile = infile.readlines()
    views = [i.rstrip() for i in infile]

    outputDir = "outputs/full-studies/"
    inputDir = "inputs/full-studies/"
    
    if not os.path.exists(outputDir):
        os.makedirs(outputDir)

    patientStudies = os.listdir(inputDir)
               
    for study in patientStudies:
            print(study)
            x = time.time()
            dcmDir = inputDir + study + "/"
            tempJPGDir = outputDir + study + '/jpg_frames/'
            resultDir = outputDir + study + '/results/'

            # create results and temporary jpg directory
            if not os.path.exists(resultDir):
                os.makedirs(resultDir)
            if os.path.exists(tempJPGDir):
                rmtree(tempJPGDir)
                os.makedirs(tempJPGDir)
            else:
                os.makedirs(tempJPGDir)
                
            # open file to save predictions
            out = open(resultDir + study + "_individual_probabilities.txt", 'w')
            out.write("study" + "\t" + "image")
            for j in views:
                out.write("\t" + "prob_" + j)
            out.write("\n")
        
            # extract frames from dcms
            extract_imgs_from_dicom(dcmDir, tempJPGDir)
            # deploy model, make predictions, delete model endpoint
            predictions = classify(tempJPGDir, predictor)

            # write probabilities each jpg to txt
            predictprobdict = {}
            for image in predictions.keys():
                prefix = image.split(".dcm")[0]
                if not predictprobdict.__contains__(prefix):
                    predictprobdict[prefix] = []
                predictprobdict[prefix].append(predictions[image])
            for prefix in predictprobdict.keys():
                out.write(dcmDir + "\t" + prefix)
                for i in predictprobdict[prefix][0]:
                    out.write("\t" + str(i))
                out.write("\n")

            out.close()
           
            y = time.time()

            print("Time:  " + str(y - x) + " seconds for " +  str(len(predictprobdict.keys()))  + " DCM frames")
            total_dcm += len(predictprobdict.keys())
            
    # Delete model endpoint
    sagemaker.Session().delete_endpoint(predictor.endpoint)
    
    total_y = time.time()
    print("Total time:  ", str(total_y - total_x))
    print("Total number DCM frames: ", str(total_dcm))

if __name__ == '__main__':
    main()
