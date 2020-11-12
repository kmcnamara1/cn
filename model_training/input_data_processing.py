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
sys.path.append('support_funcs')
from shutil import rmtree, copyfile
from echoanalysis_tools import output_imgdict, output_imgdict_still, read_dicom, read_dicom_still, extract_imgs_from_dicom

def main():

    inputDir = "inputs/" # LOCAL INPUT DIRECTORY
    outputDir = "outputs/" # LOCAL OUTPUT DIRECTORY
    inputDirList = os.listdir(inputDir) 
    
    if not os.path.exists(outputDir):
        os.makedirs(outputDir)
        
    for classLabel in inputDirList:
        inputClassDir = inputDir + classLabel + "/"
        outputClassDir = outputDir + classLabel + "/"

        if not os.path.exists(outputClassDir):
            os.makedirs(outputClassDir)
        
        extract_imgs_from_dicom(inputClassDir, outputClassDir)
        
if __name__ == '__main__':
    main()
    
