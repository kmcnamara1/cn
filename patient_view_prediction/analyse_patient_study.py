from __future__ import division, print_function, absolute_import
import numpy as np
import tensorflow as tf
import sagemaker
import random
import sys
import cv2
import pydicom
import os
# sys.path.append('echocv/funcs')
sys.path.append('support_funcs')
import time
from sagemaker.tensorflow import TensorFlowModel
from sagemaker import get_execution_role
from shutil import rmtree
from scipy.misc import imread
from echoanalysis_tools import output_imgdict, output_imgdict_still
from statistics import mean

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

def read_dicom(out_directory, filename, counter):
    if counter < 50:
        outrawfilename = filename + "_raw"
        print(filename, counter, "trying [cine]")
        rawfilenamepath = os.path.join(out_directory, outrawfilename)
        
        if os.path.exists(rawfilenamepath):
            time.sleep(2)
            
            try:
                ds = pydicom.read_file(os.path.join(out_directory, outrawfilename), force=True)
                framedict = output_imgdict(ds)
                y = len(framedict.keys()) - 1
                
                if y > 10:
                    m = random.sample(range(0, y), 10)
                    for n in m:
                        targetimage = framedict[n]
                        
                        if (n < 10):
                            outfile = os.path.join(out_directory, filename) + "_0" + str(n) + '.jpg'
                        
                        else:
                            outfile = os.path.join(out_directory, filename) + "_" + str(n) + '.jpg'
                        
                        resizedimg = cv2.resize(targetimage, (224, 224))
                        cv2.imwrite(outfile, resizedimg, [cv2.IMWRITE_JPEG_QUALITY, 95])
                        counter = 50
            
            except (IOError, EOFError, KeyError) as e:
                print(out_directory + "\t" + outrawfilename + "\t" +
                      "error", counter, e)
        
        else:
            counter = counter + 1
            time.sleep(3)
            read_dicom(out_directory, filename, counter)
    return counter


def read_dicom_still(out_directory, filename, counter):
    if counter < 50:
        outrawfilename = filename + "_raw"
        print(filename, counter, "trying [still]")
        rawfilenamepath = os.path.join(out_directory, outrawfilename)
        if os.path.exists(rawfilenamepath):
            time.sleep(2)
            
            try:
                ds = pydicom.read_file(os.path.join(out_directory, outrawfilename), force=True)
                framedict = output_imgdict_still(ds)
                targetimage = framedict[0]
                outfile = os.path.join(out_directory, filename) + "_01" + '.jpg'
                resizedimg = cv2.resize(targetimage, (224, 224))
                cv2.imwrite(outfile, resizedimg, [cv2.IMWRITE_JPEG_QUALITY, 95])
                counter = 50
            
            except (IOError, EOFError, KeyError) as e:
                print(out_directory + "\t" + outrawfilename + "\t" +
                      "error", counter, e)
        
        else:
            counter = counter + 1
            time.sleep(3)
            read_dicom_still(out_directory, filename, counter)
    return counter

def extract_imgs_from_dicom(directory, out_directory):
    """
    Extracts jpg images from DCM files in the given directory.
    Calls 'read_dicom_still' for still DCM
    Calls 'read_dicom' for cine DCM

    @param directory: folder with DCM files of echos
    @param out_directory: destination folder to where converted jpg files are placed
    @param target: destination folder to where converted jpg files are placed
    """
    allfiles = os.listdir(directory)
    for filename in allfiles[:]:
#         if not "file" in filename:
             if not "results" in filename:
                 if not "ipynb" in filename:
                    ds = pydicom.read_file(os.path.join(directory, filename),force=True)
                    
                    if ("NumberOfFrames" in  dir(ds)) and (ds.NumberOfFrames>1): #if cine
                        outrawfilename = filename + "_raw"
                        out_directory_path = out_directory + '/' + outrawfilename
                        ds.save_as(out_directory_path)
                        counter = 0
                        while counter < 5:
                            counter = read_dicom(out_directory, filename, counter)
                            counter = counter + 1
                    
                    elif (ds[0x8,0x8][3] == "0001"): # if still frame with measurements
                        outrawfilename = filename + "_raw"
                        out_directory_path = out_directory + '/' + outrawfilename
                        ds.save_as(out_directory_path)
                        counter = 0
                        while counter < 5:
                            counter = read_dicom_still(out_directory, filename, counter)
                            counter = counter + 1 
                    
                    else: # else pulse wave or M mode or Doppler?
                        print(filename + " not cine or still")
    return 1


def classify(directory, feature_dim, label_dim, model_path, role):
    """
    Classifies echo images in given directory
    Loads TensorFlowModel
    Makes predictions
    Deletes model endpoint
    
    @param directory: folder with jpg echo images for classification
    """
    # Load model
    model = TensorFlowModel(model_data=model_path, role=role, framework_version='2.1.0')
    # Deploy model
    predictor = model.deploy(initial_instance_count=1, instance_type='ml.m5.2xlarge')

    imagedict = {}
    predictions = {}
    for filename in os.listdir(directory):
        if "jpg" in filename:
            image = load_img(directory + filename, target_size=(224, 224))
            imagedict[filename] = np.array(image).reshape((1, 224, 224, 3))
        
    for filename in imagedict:
        result = predictor.predict({'inputs_input': imagedict[filename]})
        predictions[filename] = result["predictions"][0]
    
    # Delete model endpoint
    sagemaker.Session().delete_endpoint(predictor.endpoint)
    
    return predictions


def main():
    total_x = time.time()
    total_dicoms = 0
    
    role = get_execution_role()
    bucket = "sagemaker-ap-southeast-2-611188727347"
    key = "tensorflow-training-2020-11-09-01-37-28-348"
    model_path = "s3://{}/{}/output/model.tar.gz".format(bucket, key)
    
    infile = open("model_training/class_labels.txt")  #class labels
    infile = infile.readlines()
    views = [i.rstrip() for i in infile]
    
    feature_dim = 1
    label_dim = len(views)
    
    outputs_directory = "outputs/full-studies/"
    inputs_directory = "inputs/full-studies/"
    
    patient_studies = os.listdirinputs_directory)
    
    if not os.path.exists(outputs_directory):
        os.makedirs(outputs_directory)
               
    for study in patient_studies:
        if study == "bend1":
            print(study)
            x = time.time()
            dicomdir = inputs_directory + study + "/"
            temp_jpg_directory = outputs_directory + study + '/jpg_frames/'
            results_directory = outputs_directory + study + '/results/'

            # create results and temporary jpg directory
            if not os.path.exists(results_directory):
                os.makedirs(results_directory)
            if os.path.exists(temp_jpg_directory):
                rmtree(temp_jpg_directory)
                os.makedirs(temp_jpg_directory)
            
            # open file to save predictions
            out = open(results_directory + study + "_individual_probabilities_c8.txt", 'w')
            out.write("study\timage")
            for j in views:
                out.write("\t" + "prob_" + j)
            out.write('\n')
        
            # extract 10 frames from dcms
            extract_imgs_from_dicom(dicomdir, temp_jpg_directory)
            # deploy model, make predictions, delete model endpoint
            predictions = classify(temp_jpg_directory, feature_dim, label_dim, model_path, role)

            # write probabilities each jpg to txt
            predictprobdict = {}
            for image in predictions.keys():
                prefix = image.split(".dcm")[0] + ".dcm"
                if not predictprobdict.__contains__(prefix):
                    predictprobdict[prefix] = []
                predictprobdict[prefix].append(predictions[image][0])
            for prefix in predictprobdict.keys():
                predictprobmean =  np.mean(predictprobdict[prefix], axis = 0)
                out.write(dicomdir + "\t" + prefix)
                for i in predictprobmean:
                    out.write("\t" + str(i))
                out.write( "\n")

            out.close()
           
            y = time.time()

            print("time:  " + str(y - x) + " seconds for " +  str(len(predictprobdict.keys()))  + " dcms")
            total_vids += len(predictprobdict.keys())
            
    
    total_y = time.time()
    print("total time:  ", str(total_y - total_x))
    print("total number dcms: ", str(total_dicoms))

if __name__ == '__main__':
    main()
