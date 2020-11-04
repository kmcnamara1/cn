# coding: utf-8
import sys
import os
import pydicom
import time
# sys.path.append("/home/rdeo/anaconda/lib/python2.7/site-packages/")
import numpy as np
import subprocess
from subprocess import Popen, PIPE
from scipy.misc import imresize
import cv2

def computehr_gdcm(ds):
    hr = "None"
    hr = ds[0x18,0x1088].value
    return int(hr)

def computexy_gdcm(ds):
    rows = ds[0x28,0x10].value
    cols = ds[0x28,0x11].value
    return int(rows), int(cols)

def computebsa_gdcm(ds):
    '''
    dubois, height in m, weight in kg
    :param data: 
    :return: 
    '''
    h = ds[0x10,0x1020].value
    w = ds[0x10,0x1030].value
    return 0.20247 * (eval(h)**0.725) * (eval(w)**0.425)

def computedeltaxy_gdcm(ds):
    '''
    the unit is the number of cm per pixel 
    '''
    xlist = []
    ylist = []
    xlist.append(ds[0x18,0x6011][0][0x18,0x602c].value)
    ylist.append(ds[0x18,0x6011][0][0x18,0x602e].value)
    return np.min(xlist), np.min(ylist)

def remove_periphery(imgs):
    imgs_ret = []
    for img in imgs:
        image = img.astype('uint8').copy()
        fullsize = image.shape[0] * image.shape[1]
        image[image > 0 ] = 255
        image = cv2.bilateralFilter(image, 11, 17, 17)
        thresh = cv2.threshold(image, 200, 255, cv2.THRESH_BINARY)[1]
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        contours = cnts[1]
        areas = []
        for i in range(0, len(contours)):
            areas.append(cv2.contourArea(contours[i]))

        if len(areas) == 0:
            imgs_ret.append(img)
        else:
            select = np.argmax(areas)
            roi_corners_clean = []
            roi_corners = np.array(contours[select], dtype = np.int32)
            for i in roi_corners:
                roi_corners_clean.append(i[0])
            hull = cv2.convexHull(np.array([roi_corners_clean], dtype = np.int32))
            mask = np.zeros(image.shape, dtype=np.uint8)
            mask = cv2.fillConvexPoly(mask, hull, 1)
            imgs_ret.append(img*mask)
    return np.array(imgs_ret)

def computeft_gdcm(video, study, appdir):
    videodir = appdir + "static/studies/" + study.file
    command = 'gdcmdump ' + videodir + "/" + video.file + "| grep Frame"
    pipe = Popen(command, stdout=PIPE, stderr=None, shell=True)
    text = pipe.communicate()[0]
    data = text.split("\n")
    defaultframerate = 30
    counter = 0
    for i in data:
        if i.split(" ")[0] == '(0018,1063)':
            frametime = i.split(" ")[2][1:-1]
            counter = 1
        elif i.split(" ")[0] == '(0018,0040)':
            framerate = i.split("[")[1].split(" ")[0][:-1]
            frametime = str(1000 / eval(framerate))
            counter = 1
        elif i.split(" ")[0] == '(7fdf,1074)':
            framerate = i.split(" ")[3]
            frametime = str(1000 / eval(framerate))
            counter = 1
    if not counter == 1:
        print("missing framerate")
        framerate = defaultframerate
        frametime = str(1000 / framerate)
    ft = eval(frametime)
    return ft

def computeft_gdcm_strain(ds):
    defaultframerate = None
    frametime = ds[0x18,0x1063].value #frametime
    framerate = ds[0x18,0x40].value
    ft = frametime
    return int(ft)

def output_imgdict(imagefile):
    '''
    converts raw dicom to numpy arrays
    '''
    try:
        ds = imagefile
        if len(ds.pixel_array.shape) == 4: 
            #format: nframes, nrows, ncols, YUV
            nframes = ds.pixel_array.shape[0]
            maxframes = nframes * 3
        nrow = int(ds.Rows)
        ncol = int(ds.Columns)
        ArrayDicom = np.zeros((nrow, ncol), dtype=ds.pixel_array.dtype)
        imgdict = {}
        for counter in range(0, nframes):
            a = ds.pixel_array[counter,:,:,0]
            g = a.reshape(1, nrow * ncol)
            y = g.reshape(nrow, ncol)
            u = np.zeros((nrow, ncol), dtype=ds.pixel_array.dtype)
            v = np.zeros((nrow, ncol), dtype=ds.pixel_array.dtype)
            ArrayDicom[:, :] = ybr2gray(y,u,v)
            ArrayDicom[0:int(nrow / 10), 0:int(ncol)] = 0  # blanks out name
            ArrayDicom.clip(0)
            nrowout = nrow
            ncolout = ncol
            x = int(counter)
            imgdict[x] = imresize(ArrayDicom, (nrowout, ncolout))
        return imgdict
    except:
        print("Unexpected error:", sys.exc_info()[0])
        pass
    
def output_imgdict_still(imagefile):
    '''
    converts raw dicom to numpy arrays
    '''
    try:
        ds = imagefile
        if len(ds.pixel_array.shape) == 3: #format nrow, ncol, depth
            nframes = 1
            maxframes = nframes * 1
        nrow = int(ds.Rows)
        ncol = int(ds.Columns)
        ArrayDicom = np.zeros((nrow, ncol), dtype=ds.pixel_array.dtype)
        imgdict = {}
        for counter in range(0, nframes):
            a = ds.pixel_array[:,:,0]
            g = a.reshape(1, nrow * ncol)
            y = g.reshape(nrow, ncol)
            u = np.zeros((nrow, ncol), dtype=ds.pixel_array.dtype)
            v = np.zeros((nrow, ncol), dtype=ds.pixel_array.dtype)
            ArrayDicom[:, :] = ybr2gray(y,u,v)
            ArrayDicom[0:int(nrow / 10), 0:int(ncol)] = 0  # blanks out name
            ArrayDicom.clip(0)
            nrowout = nrow
            ncolout = ncol
            x = int(counter)
            imgdict[x] = imresize(ArrayDicom, (nrowout, ncolout))
        return imgdict
    except:
        print("Unexpected error:", sys.exc_info()[0])
        pass


def create_mask(imgs):
    '''
    removes static burned in pixels in image; will use for disease diagnosis
    '''
    from scipy.ndimage.filters import gaussian_filter
    diffs = []
    for i in range(len(imgs) - 1):
        temp = np.abs(imgs[i] - imgs[i + 1])
        temp = gaussian_filter(temp, 10)
        temp[temp <= 50] = 0
        temp[temp > 50] = 1

        diffs.append(temp)

    diff = np.mean(np.array(diffs), axis=0)
    diff[diff >= 0.5] = 1
    diff[diff < 0.5] = 0
    return diff

def ybr2gray(y, u, v):
    r = y + 1.402 * (v - 128)
    g = y - 0.34414 * (u - 128) - 0.71414 * (v - 128)
    b = y + 1.772 * (u - 128)
    # print r, g, b
    gray = (0.2989 * r + 0.5870 * g + 0.1140 * b)
    return np.array(gray, dtype="int8")


def create_imgdict_from_dicom(directory, filename):
    """
    convert compressed DICOM format into numpy array
    """
    targetfile = os.path.join(directory, filename)
    # temp_directory = os.path.join(directory, "image_segment")
    # if not os.path.exists(temp_directory):
    #     os.makedirs(temp_directory)
    ds = pydicom.read_file(targetfile, force = True)
    if ("NumberOfFrames" in  dir(ds)) and (ds.NumberOfFrames>1):
        # outrawfile = os.path.join(temp_directory, filename + "_raw")
        # command = 'gdcmconv -w ' + os.path.join(directory, filename) + " "  + outrawfile
        # subprocess.Popen(command, shell=True)
        # time.sleep(10)
        if os.path.exists(targetfile):
            ds = pydicom.read_file(targetfile, force = True)
            imgdict = output_imgdict(ds)
        else:
            print(targetfile, "missing")
    return imgdict


