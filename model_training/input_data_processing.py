from __future__ import division, print_function, absolute_import
import numpy as np
import tensorflow as tf
import random
import sys
import cv2
import pydicom
import os
sys.path.append('echocv/funcs')
sys.path.append('echocv/nets')
import subprocess
import time
from shutil import rmtree
from shutil import copyfile
from optparse import OptionParser
from scipy.misc import imread
from echoanalysis_tools import output_imgdict, output_imgdict_still
from matplotlib import pyplot as plt
from statistics import mean

def read_dicom(out_directory, filename, counter):
    if counter < 50:
        outrawfilename = filename + "_raw"
        print(filename, counter, "trying (cine)")
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
            print("D")
            read_dicom(out_directory, filename, counter)
    return counter


def read_dicom_still(out_directory, filename, counter):
    if counter < 50:
        outrawfilename = filename + "_raw"
        print(filename, counter, "trying (still)")
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

def extract_imgs_from_dicom(in_directory, out_directory):
    """
    Extracts jpg images from DCM files in the given in_directory

    @param in_directory: folder with DCM files of echos
    @param out_directory: destination folder to where converted jpg files are placed
    @param target: destination folder to where converted jpg files are placed
    """
    allfiles = os.listdir(in_directory)
    for filename in allfiles[:]:
             if not "results" in filename:
                if not "ipynb" in filename:
                    ds = pydicom.read_file(os.path.join(in_directory, filename),force=True)
                    if ("NumberOfFrames" in  dir(ds)) and (ds.NumberOfFrames>1): #if cine
                        outrawfilename = filename + "_raw"
                        out_directory_path = out_directory + '/' + outrawfilename
                        ds.save_as(out_directory_path)
                        counter = 0
                        while counter < 5:
                            counter = read_dicom(out_directory, filename, counter)
                            counter = counter + 1
                    elif (ds[0x8,0x8][3] == "0001"): # if still with measurements
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

def main():
    total_x = time.time()
    total_dicoms = 0

    input_class_dir = os.listdir("inputs/")
    output_dir = "outputs/"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    for class_label in input_class_dir:
            x = time.time()
            dicomdir = "inputs/" + class_label + "/"
            output_class_dir = output_dir + class_label + "/"

            if not os.path.exists(output_class_dir):
                os.makedirs(output_class_dir)
            
            num_dicoms = len(dicomdir)
            total_dicoms += num_dicoms

            extract_imgs_from_dicom(dicomdir, output_class_dir)
            
            y = time.time()

            print("time:  " + str(y - x) + " seconds for " +  num_dicoms  + " DICOM files")            
    
    total_y = time.time()
    print("total time:  ", str(total_y - total_x))
    print("total number videos: ", str(total_dicoms))

if __name__ == '__main__':
    main()
