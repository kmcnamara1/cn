from __future__ import division, print_function, absolute_import
import os
import sys
import cv2
import time
import random
import pydicom
import subprocess
import numpy as np
import tensorflow as tf
from shutil import rmtree
from shutil import copyfile

def analyse_probabilities(inputPatientDir, outputPatientDir):

    inputPatientDirList = os.listdir(inputPatientDir)

    for patient in inputPatientDirList:
            filelist = os.listdir(inputPatientDir + patient + "/")
            patientResultDir = outputPatientDir + patient + '/results/'        
            out = open(patientResultDir + patient + "_study_probabilities_c8.txt", 'w')

            classDict = {}
            infile = open("model_training/class_labels.txt") 
            infile = infile.readlines()
            infile = [i.rstrip() for i in infile]
            for i in range(len(infile)):
                classDict[infile[i]] = i + 2

            probabilityFile = outputPatientDir + patient + "/results/" + patient + "_individual_probabilities_c8.txt"
            infile = open(probabilityFile)
            infile = infile.readlines()
            infile = [i.rstrip() for i in infile]
            infile = [i.split('\t') for i in infile]

            predictedClass90 = {}
            out.write("Filename" + "\t" + "Predicted Class" + "\t" + "Likelihood" + "\n")
            for key in classDict.keys():
                predictedClass90[key] = []
            for i in infile[1:]: #for each filename
                filenameOriginal = i[1]
                filename = filenameOriginal[:-7]

                for key in classDict.keys():
                    if eval(i[classDict[key]]) > 0.9:
                        predictedClass90[key].append(filename)
            
            predictedClasses = []
            for key in predictedClass90.keys():
                    predictedClassesArray = []
                    for filename in predictedClass90[key]:
                        count = predictedClass90[key].count(filename)
                        if count >= 3:
                            if filename not in predictedClassesArray:
                                predictedClassesArray.append(filename)
                                predictedClasses.append(filename)
                                out.write(filename + "\t" + str(key) + "\t" + "0.9" + "\n")
                    
            for file in filelist:
                if file not in predictedClasses:
                    out.write(file + "\t" + "none" + "\t" + "0" + "\n")
              
            out.close()


def main():
    inputDir = "inputs/full-studies/" #local directory of patient DICOMS
    outputDir = "outputs/full-studies/" #local directory of patient results
    analyse_probabilities(inputDir, outputDir)


if __name__ == '__main__':
    main()

                    
                    