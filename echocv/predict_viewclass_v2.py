from __future__ import division, print_function, absolute_import
import numpy as np
import tensorflow as tf
import random
import sys
import cv2
import pydicom
import os
sys.path.append('/content/gdrive/My Drive/CardioNexus/GitHubRepo/cn/echocv/funcs')
sys.path.append('/content/gdrive/My Drive/CardioNexus/GitHubRepo/cn/echocv/nets')
import subprocess
import time
from shutil import rmtree
from shutil import copyfile
from optparse import OptionParser
from scipy.misc import imread
from echoanalysis_tools import output_imgdict
from matplotlib import pyplot as plt

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


# # Hyperparams
parser=OptionParser()
parser.add_option("-d", "--dicomdir", dest="dicomdir", help = "dicomdir")
parser.add_option("-g", "--gpu", dest="gpu", default = "0", help = "cuda device to use")
parser.add_option("-m", "--model", dest="model")
params, args = parser.parse_args()
dicomdir = params.dicomdir
model = params.model

import vgg as network

os.environ["CUDA_VISIBLE_DEVICES"] = params.gpu

def read_dicom(out_directory, filename, counter):
    if counter < 50:
        outrawfilename = filename + "_raw"
        print(out_directory, filename, counter, "trying")
        rawfilenamepath = os.path.join(out_directory, outrawfilename)
        if os.path.exists(rawfilenamepath):
            time.sleep(2)
            try:
                ds = pydicom.read_file(os.path.join(out_directory, outrawfilename), force=True)
                framedict = output_imgdict(ds)
                y = len(framedict.keys()) - 1
                if y > 10:
                    # m = random.sample(range(0, y), 10)
                    m = range(0, y)
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
            print("D")
            read_dicom(out_directory, filename, counter)
    return counter

def extract_imgs_from_dicom(directory, out_directory):
    """
    Extracts jpg images from DCM files in the given directory

    @param directory: folder with DCM files of echos
    @param out_directory: destination folder to where converted jpg files are placed
    @param target: destination folder to where converted jpg files are placed
    """
    allfiles = os.listdir(directory)

    for filename in allfiles[:]:
      if not "image" in filename:
              print(filename)
              ds = pydicom.read_file(os.path.join(directory, filename),force=True)
              if ("NumberOfFrames" in  dir(ds)) and (ds.NumberOfFrames>1):
                  outrawfilename = filename + "_raw"
                  out_directory_path = out_directory + '/' + outrawfilename
                  ds.save_as(out_directory_path)
                  counter = 0
                  while counter < 5:
                      counter = read_dicom(out_directory, filename, counter)
                      counter = counter + 1
    return 1


def classify(directory, feature_dim, label_dim, model_name):
    """
    Classifies echo images in given directory

    @param directory: folder with jpg echo images for classification
    """
    imagedict = {}
    predictions = {}
    for filename in os.listdir(directory):
        if "jpg" in filename:
            image = imread(directory + filename, flatten = True).astype('uint8')
            imagedict[filename] = [image.reshape((224,224,1))]

    tf.reset_default_graph() #clear default graph stack and reset global default graph
    sess = tf.Session() #class for running TF operations
    model = network.Network(0.0, 0.0, feature_dim, label_dim, False)
    sess.run(tf.global_variables_initializer())

    saver = tf.train.Saver() #save and restore variables
    saver.restore(sess, model_name) #restore model

    for filename in imagedict:
        predictions[filename] =np.around(model.probabilities(sess, imagedict[filename]), decimals = 3)
    
    return predictions


def main():
    model = "view_23_e5_class_11-Mar-2018"
    dicomdir = "/content/gdrive/My Drive/CardioNexus/dicomsample/EchoCV-Test-Labelled/"
    model_name = '/content/gdrive/My Drive/CardioNexus/echocv_models/' + model
    infile = open("/content/gdrive/My Drive/CardioNexus/GitHubRepo/cn/echocv/viewclasses_" + model + ".txt")
    infile = infile.readlines()
    views = [i.rstrip() for i in infile]

    feature_dim = 1
    label_dim = len(views)

    # out = open(model + "_" + dicomdir  + "_probabilities.txt", 'w')
    out = open("/content/gdrive/My Drive/CardioNexus/GitHubRepo/cn/echocv/results/view_23_e5_class_11-Mar-2018_dicomsample_probabilities.txt", 'w')
    out.write("study\timage")
    for j in views:
        out.write("\t" + "prob_" + j)
    out.write('\n')

    x = time.time()
    temp_image_directory = dicomdir + 'image_view/'
    # if os.path.exists(temp_image_directory):
    #     rmtree(temp_image_directory)
    if not os.path.exists(temp_image_directory):
        os.makedirs(temp_image_directory)
    extract_imgs_from_dicom(dicomdir, temp_image_directory)
    predictions = classify(temp_image_directory, feature_dim, label_dim, model_name)
    
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

    y = time.time()

    print("time:  " +str(y - x) + " seconds for " +  str(len(predictprobdict.keys()))  + " videos")
    # rmtree(temp_image_directory)
    out.close()

if __name__ == '__main__':
    main()
