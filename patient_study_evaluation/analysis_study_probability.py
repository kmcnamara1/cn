from __future__ import division, print_function, absolute_import
import numpy as np
import tensorflow as tf
import random
import sys
import cv2
import pydicom
import os
import subprocess
import time
from shutil import rmtree
from shutil import copyfile
from optparse import OptionParser
from scipy.misc import imread
from matplotlib import pyplot as plt

def analyse_probabilities(input_patients_directory, output_patients_directory):

    input_patients_directory_list = os.listdir(input_patients_directory)

    for patient in input_patients_directory_list:

            filelist = os.listdir(input_patients_directory + patient + "/")
            patient_result_directory = output_patients_directory + patient + '/results/'        
            out = open(patient_result_directory + patient + "_study_probabilities.txt", 'w')

            viewdict = {}
            infile = open("viewclasses_view_23_e5_class_11-Mar-2018.txt") 
            infile = infile.readlines()
            infile = [i.rstrip() for i in infile]
            for i in range(len(infile)):
                viewdict[infile[i]] = i + 2

            viewfile = output_patients_directory + patient + "/results/" + patient + "_individual_probabilities.txt"
            infile = open(viewfile)
            infile = infile.readlines()
            infile = [i.rstrip() for i in infile]
            infile = [i.split('\t') for i in infile]

            pred_views_90 = {}
            unpred_views = {}
            out.write("filename" + "\t" + "predicted class" + "\t" + "probability" + "\n")
            for key in viewdict.keys():
                pred_views_90[key] = []
                unpred_views[key] = []
            for i in infile[1:]: #for each filename
                filename_og = i[1]
                filename = filename_og[:-11]

                for key in viewdict.keys():
                    if eval(i[viewdict[key]]) > 0.9:
                        pred_views_90[key].append(filename)
                    else:
                        unpred_views[key].append(filename)
            
            found_views = []
            for key in pred_views_90.keys():
                    out_arr = []
                    for filename in pred_views_90[key]:
                        count = pred_views_90[key].count(filename)
                        if count >= 3:
                            if filename not in out_arr:
                                out_arr.append(filename)
                                found_views.append(filename)
                                out.write(filename + "\t" + str(key) + "\t" + "0.9" + "\n")
                    
            for file in filelist:
                if file not in found_views:
                    out.write(file + "\t" + "none" + "\t" + "0" + "\n")
              
            out.close()


def main():
    input_patients_directory = "inputs/full-studies/" #local directory of patient DICOMS
    output_patients_directory = "outputs/full-studies/" #local directory of patient results

    analyse_probabilities(input_patients_directory, output_patients_directory)


if __name__ == '__main__':
    main()

                    
                    