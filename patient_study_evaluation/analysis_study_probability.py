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

viewdirs = os.listdir("inputs/full-studies/")
outputs_directory = "outputs/full-studies/"

def analyse_probabilities():
    for study in viewdirs:
            filelist = os.listdir("inputs/full-studies/" + study + "/")
            results_directory = outputs_directory + study + '/results/'        
            out = open(results_directory + study + "_study_probabilities.txt", 'w')

            viewdict = {}
            infile = open("echocv/viewclasses_view_23_e5_class_11-Mar-2018.txt") 
            infile = infile.readlines()
            infile = [i.rstrip() for i in infile]
            for i in range(len(infile)):
                viewdict[infile[i]] = i + 2

            viewfile = "outputs/full-studies/" + study + "/results/" + study + "_individual_probabilities.txt"
            infile = open(viewfile)
            infile = infile.readlines()
            infile = [i.rstrip() for i in infile]
            infile = [i.split('\t') for i in infile]

            pred_views_90 = {}
            unpred_views = {}
            out.write("filename" + "\t" + "predicted view" + "\t" + "val" + "\n")
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
                    out_arr2 = []
                    for filename in pred_views_90[key]:
                        count = pred_views_90[key].count(filename)
                        if count >= 3:
                            if filename not in out_arr:
                                out_arr.append(filename)
                                found_views.append(filename)
                                out.write(filename + "\t" + str(key) + "\t" + "0.9" + "\n")
                    
            for file in filelist:
                if file not in found_views:
                    out.write(file + "\t" + "None" + "\t" + "0" + "\n")
              
            out.close()


def main():
    analyse_probabilities()

if __name__ == '__main__':
    main()

                    
                    