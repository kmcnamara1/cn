from __future__ import division, print_function, absolute_import
import numpy as np
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
import random
import sys, os, time, json
import math
from easydict import EasyDict as edicton
sys.path.append('src/')
from util import *
from hmmlearn import hmm
from scipy.misc import imread, imresize, imsave
from scipy.ndimage.filters import gaussian_filter
from skimage import exposure
from scipy.signal import convolve
from optparse import OptionParser
import csv 
import scipy.interpolate as interp 
import copy

tf.compat.v1.disable_eager_execution()

parser=OptionParser()
parser.add_option("-g", "--gpu", dest="gpu", default = "0", help = "cuda device to use")
params, args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = params.gpu

print("start")
def bridgeAcross(trace, frac):
    classList = range(6)
    newTrace = copy.copy(trace)
    windowLength = 10
    for i in range(0, len(trace), 2):
        window = newTrace[i:(i+windowLength)] 
        for choice in classList:
            if np.sum(window==choice) > frac*windowLength:
                for k in range(2,10):
                    testWindow = newTrace[i+k*windowLength:(i+(k+1)*windowLength)] 
                    midWindow = newTrace[i+windowLength:(i+(k)*windowLength)] 
                    if np.sum(testWindow==choice) > frac*windowLength:
                        goodClass = np.sum(midWindow==(choice))
                        if goodClass < len(midWindow):
                            print(goodClass, len(midWindow), i)
                            end = min(len(trace), i + (k+1)*windowLength)
                            for j in range(i, end):
                                newTrace[j] = choice 
    return newTrace

def bridgeUp(trace, frac):
    classList = range(6)
    newTrace = copy.copy(trace)
    windowLength = 10
    for i in range(0, len(trace), 2):
        window = newTrace[i:(i+windowLength)] 
        for choice in classList:
            if choice == 5:
                nextStep = 0
            else:
                nextStep = choice + 1
            if np.sum(window==choice) > frac*windowLength:
                for k in range(2, 10):
                    testWindow = newTrace[i+k*windowLength:(i+(k+1)*windowLength)] 
                    midWindow = newTrace[i+windowLength:(i+(k)*windowLength)] 
                    if np.sum(testWindow==(nextStep)) > frac*windowLength:
                        goodClass = np.sum(midWindow==(nextStep)) + np.sum(midWindow==(choice))
                        if goodClass < len(midWindow):
                            print(goodClass, len(midWindow), i)
                            end = min(len(trace), i + (k+1)*windowLength)
                            for j in range(i, end):
                                if not newTrace[j] == nextStep:
                                   if not newTrace[j] == choice: 
                                        newTrace[j] = choice 
    return newTrace

def bridgeAcrossFinal(trace, frac):
    classList = range(6)
    newTrace = copy.copy(trace)
    windowLength = 15
    for i in range(0, len(trace), int(0.5*windowLength)):
        window = newTrace[i:(i+windowLength)] 
        for choice in classList:
            if np.sum(window==choice) > frac*windowLength:
                for k in range(2,6):
                    end = min(len(trace), i + (k+1)*windowLength)
                    testWindow = newTrace[i+k*windowLength:end] 
                    if np.sum(testWindow==choice) > frac*windowLength:
                        for j in range(i, end):
                            newTrace[j] = choice
    return newTrace



def load_model(config, sess):
    return Unet(config, sess)

def smoothRows(rows):
    box_pts = 20 #choice of window 
    box = np.ones((1,1,box_pts,1))/float(box_pts)
    y_smooth = convolve(rows, box, mode='same')
    return y_smooth.astype(int)
    
    
def train_print(i, j, loss, batch, batch_total, time):
    '''
    Formats print statements to update on same print line.
    
    @params are integers or floats
    '''
    print("Epoch {:1} |".format(i), 
          "Iter {:1} |".format(j), 
          "Loss: {:.4} |".format(loss),
          "Data: {}/{} |".format(batch, batch_total), 
          "Time {:1.2} ".format(time), 
          "   ", end="\r")
    
def val_print(i, j, loss, acc, time):
    '''
    Formats print statements to update on same print line.
    
    @params are integers or floats
    '''
    print("Epoch {:1} |".format(i), 
          "Iter {:1} |".format(j), 
          "Loss: {:.2} |".format(loss),
          "Acc: {} |".format(np.round(acc,3)), 
          "Time {:1.2} ".format(time), 
          "   ", end="\r")
    

class Unet(object):        
    def __init__(self, config, sess):
        self.config = config
        self.sess = sess
        self.x_train = tf.compat.v1.placeholder(tf.float32, [None, 12, 2000, config.feature_dim])
        self.y_train = tf.compat.v1.placeholder(tf.float32, [None, 1, 2000, config.label_dim])
        self.x_test = tf.compat.v1.placeholder(tf.float32, [None, 12, 2000, config.feature_dim])
        self.y_test = tf.compat.v1.placeholder(tf.float32, [None, 1, 2000, config.label_dim])
        
        self.global_step = tf.Variable(0, trainable=False)
        self.output = self.unet(self.x_train, config.mean, keep_prob = config.dropout)
        self.loss = (tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = self.output, labels = self.y_train)) 
            + config.weight_decay * tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'batch_normalization' not in v.name]))
        self.opt = tf.train.AdamOptimizer(config.learning_rate).minimize(self.loss, global_step = self.global_step)
        
        self.pred = self.unet(self.x_test, config.mean, keep_prob = 1.0, reuse = True)
        self.loss_summary = tf.summary.scalar('loss', self.loss)

  
    def train(self, x_train, y_train, x_test, y_test, saver, summary_writer, checkpoint_path, val_output_dir):
        sess = self.sess
        config = self.config
        batch_size = config.batch_size
        losses = deque([])
        train_accs = deque([])
        step = tf.train.global_step(sess, self.global_step)
        for i in range(config.epochs):
            # Shuffle indicies
            indicies = range(x_train.shape[0])
            np.random.shuffle(indicies)
            # Start timer
            start = timeit.default_timer()

            for j in range(int(x_train.shape[0]/batch_size)):
                temp_indicies = indicies[j*batch_size:(j+1)*batch_size]
                loss, loss_summary = self.fit_batch(x_train[temp_indicies], y_train[temp_indicies])
                
                if step % config.summary_interval == 0:
                    summary_writer.add_summary(loss_summary, step)
                if len(losses) == config.loss_smoothing:
                    losses.popleft()
                losses.append(loss)

                stop = timeit.default_timer()
                train_print(i, j, np.mean(losses), j*batch_size, x_train.shape[0], stop - start)
                step = step + 1

            stop = timeit.default_timer()
            acc = self.validate(x_test, y_test)
            summary = tf.Summary()
            for k in range(len(acc)):
                summary.value.add(tag="validation_acc_" + str(k), simple_value=acc[k])
            if summary_writer:    
                summary_writer.add_summary(summary, step)
            val_print(i, j, np.mean(losses), acc, stop - start)
            print()

            if (i+1) % config.epoch_save_interval == 0:
                saver.save(sess, checkpoint_path, global_step=step)
                self.visualize(x_test, y_test, val_output_dir)
        if (i+1) % config.epoch_save_interval != 0:
            saver.save(sess, checkpoint_path, global_step=step)
            self.visualize(x_test, y_test, val_output_dir)

        return True


    def validate(self, x_test, y_test):
        '''
        Calculates accuracy of validation set
        
        @params sess: Tensorflow Session
        @params x_test: Numpy array of validation images
        @params y_test: Numpy array of validation labels
        @params batch_size: Integer defining mini-batch size
        '''

        scores = [0] * int(y_test.shape[3])
        counts = [0] * int(y_test.shape[3])
        for i in range(int(x_test.shape[0])):
            gt = np.argmax(y_test[i,:,:,:], 2)
            pred = np.argmax(self.predict(x_test[i:i+1])[0,:,:,:], 2)
            for j in range(int(y_test.shape[3])):
                dice = iou(gt, pred, j)
                if not math.isnan(dice):
                    scores[j] = scores[j] + dice
                    counts[j] = counts[j] + 1
                
        return [score/counts[i] for i, score in enumerate(scores)]

    def fit_batch(self, x_train, y_train):
        _, loss, loss_summary = self.sess.run((self.opt, self.loss, self.loss_summary), feed_dict={self.x_train: x_train, self.y_train: y_train})
        return loss, loss_summary
    
    def predict(self, x):
        prediction = self.sess.run((self.pred), feed_dict={self.x_test: x})
        return prediction

    def visualize(self, x_test, y_test, val_output_dir):
        out_dir = val_output_dir + '-' + str(self.sess.run(self.global_step))
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        for i in range(x_test.shape[0]):
            gt = np.argmax(y_test[i,:,:,:], 2)[0]
            pred = np.argmax(self.predict(x_test[i:i+1])[0,:,:,:], 2)[0]
            plt.figure(figsize = (12,4))
            plt.subplot(121)
            plt.plot(pred)
            plt.subplot(122)
            plt.plot(gt)
            out_filename = 'output' + str(i) + '.png'
            plt.savefig(os.path.join(out_dir,out_filename))
            plt.close()


    def unet(self, input, mean, keep_prob = 0.5, reuse = None):
        config = self.config
        with tf.compat.v1.variable_scope('vgg', reuse=reuse):
            input = input - mean
            pool_ = lambda x: max_pool(x, 2, 2)
            pool_rect_ = lambda x: max_pool_rect(x, (1,2), (1,2))
            conv_ = lambda x, output_depth, name, padding = 'SAME', relu = True, filter_size = 3: conv(x, filter_size, output_depth, 1, config.weight_decay, name=name, padding=padding, relu=relu)
            conv_rect_ = lambda x, output_depth, name, padding = 'SAME', relu = True, filter_size = (3,3): conv_rect(x, filter_size, output_depth, 1, config.weight_decay, name=name, padding=padding, relu=relu)

            deconv_ = lambda x, output_depth, name: deconv(x, 2, output_depth, 2, config.weight_decay, name=name)
            deconv_rect_ = lambda x, output_depth, name, size=2: deconv(x, (1,size), output_depth, (1,size), config.weight_decay, name=name)

            fc_ = lambda x, features, name, relu = True: fc(x, features, config.weight_decay, name, relu)
            
            conv_1_1 = conv_rect_(input, 64, 'conv1_1', filter_size = (1,config.filter_size_1))
            conv_1_2 = conv_rect_(conv_1_1, 64, 'conv1_2',  filter_size = (1,config.filter_size_1))
            conv_1_3 = conv_rect_(conv_1_2, 64, 'conv1_3',  filter_size = (1,config.filter_size_1))

            pool_1 = pool_rect_(conv_1_3)
            
            conv_2_1 = conv_rect_(pool_1, 128, 'conv2_1', filter_size = (1,config.filter_size_1))
            conv_2_2 = conv_rect_(conv_2_1, 128, 'conv2_2', filter_size = (1,config.filter_size_1))
            conv_2_3 = conv_rect_(conv_2_2, 128, 'conv2_3', filter_size = (1,config.filter_size_1))
            
            pool_2 = pool_rect_(conv_2_3)
      
            conv_3_1 = conv_rect_(pool_2, 256, 'conv3_1', filter_size = (1,config.filter_size_1))
            conv_3_2 = conv_rect_(conv_3_1, 256, 'conv3_2', filter_size = (1,config.filter_size_1))
            conv_3_3 = conv_rect_(conv_3_2, 256, 'conv3_3', filter_size = (1,config.filter_size_1))
            
            pool_3 = pool_rect_(conv_3_3)
            
            conv_4_1 = conv_rect_(pool_3, 512, 'conv4_1',  filter_size = (1,config.filter_size_2))
            conv_4_2 = conv_rect_(conv_4_1, 512, 'conv4_2', filter_size = (1,config.filter_size_2))
            conv_4_3 = conv_rect_(conv_4_2, 512, 'conv4_3', filter_size = (1,config.filter_size_2))

            pool_4 = pool_rect_(conv_4_3)
            
            conv_5_1 = conv_rect_(pool_4, 1024, 'conv5_1', filter_size = (1,config.filter_size_2))
            conv_5_2 = conv_rect_(conv_5_1, 1024, 'conv5_2', filter_size = (1,config.filter_size_2))
            conv_5_3 = conv_rect_(conv_5_2, 1024, 'conv5_3', filter_size = (1,config.filter_size_2))
            
            pool_5 = tf.concat([deconv_rect_(conv_5_3, 512, 'up5', 8), conv_2_3], 3)
            
            conv_6_1 = conv_rect_(pool_5, 512, 'conv6_1', filter_size = (1,config.filter_size_2))
            conv_6_2 = conv_rect_(conv_6_1, 512, 'conv6_2', filter_size = (1,config.filter_size_2))
            conv_6_3 = conv_rect_(conv_6_2, 512, 'conv6_3', filter_size = (1,config.filter_size_2))
  
            deconv_6 = tf.concat([deconv_rect_(conv_6_3, 256, 'up6'), conv_1_3], 3)
            
            conv_7_1 = conv_rect_(deconv_6, 256, 'conv7_1',  filter_size = (1,config.filter_size_3))
            conv_7_2 = conv_rect_(conv_7_1, 256, 'conv7_2', filter_size = (1,config.filter_size_3))
            conv_7_3 = conv_rect_(conv_7_2, 256, 'conv7_3', filter_size = (1,config.filter_size_3))
            
            conv_8_0 = conv_rect_(conv_7_3, 512, 'conv8_0', filter_size = (12,1), padding='VALID')
            conv_8_1 = conv_rect_(conv_8_0, 512, 'conv8_1',  filter_size = (1,config.filter_size_3))
            conv_8_2 = conv_rect_(conv_8_1, 512, 'conv8_2', filter_size = (1,config.filter_size_3))
            conv_8_3 = conv_rect_(conv_8_2, 512, 'conv8_3', filter_size = (1,config.filter_size_3))
            
            pool_8 = pool_rect_(conv_8_3)
            
            conv_9_1 = conv_rect_(pool_8, 1024, 'conv9_1', filter_size = (1,config.filter_size_3))
            conv_9_2 = conv_rect_(conv_9_1, 1024, 'conv9_2', filter_size = (1,config.filter_size_3))
            conv_9_3 = conv_rect_(conv_9_2, 1024, 'conv9_3', filter_size = (1,config.filter_size_3))
            
            deconv_9 = tf.concat([deconv_rect_(conv_9_3, 512, 'up9'), conv_8_3], 3)  
            
            conv_10_1 = tf.nn.dropout(conv_rect_(deconv_9, 512, 'conv10_1', filter_size = (1,config.filter_size_4)), keep_prob)
            conv_10_2 = tf.nn.dropout(conv_rect_(conv_10_1, 512, 'conv10_2', filter_size = (1,config.filter_size_4)), keep_prob)
            conv_10_3 = tf.nn.dropout(conv_rect_(conv_10_2, 512, 'conv10_3', filter_size = (1,config.filter_size_4)), keep_prob)
            
            conv_11 = conv_(conv_10_3, config.label_dim, 'conv11', filter_size = 1, relu = False)
            return conv_11

    def init_weights(self):
        pass

def smoothArray(oldArray): 
    newArray = np.apply_along_axis(smoothRow, 1, oldArray) 
    return newArray

def expandLead(oldLead): 
    arr_interp = interp.interp1d(np.arange(oldLead.shape[0]), oldLead, kind="nearest") 
    arr_stretch = arr_interp(np.linspace(0,oldLead.shape[0]-1, 10000)) 
    return arr_stretch

def resizeLead(oldLead, targetSize): 
    origSize = oldLead.shape[0] 
    arr_interp = interp.interp1d(np.arange(origSize), oldLead) 
    arr_stretch = arr_interp(np.linspace(0, origSize-1, targetSize)) 
    return arr_stretch

def rle(inarray):
    """ run length encoding. Partial credit to R rle function. 
        Multi datatype arrays catered for including non Numpy
        returns: tuple (runlengths, startpositions, values) """
    ia = np.asarray(inarray)                  # force numpy
    n = len(ia)
    if n == 0: 
        return (None, None, None)
    else:
        y = np.array(ia[1:] != ia[:-1])     # pairwise unequal (string safe)
        i = np.append(np.where(y), n - 1)   # must include last element posi
        z = np.diff(np.append(-1, i))       # run lengths
        p = np.cumsum(np.append(0, z))[:-1] # positions
        return(z, p, ia[i])


def load_data(config, study_num):
    directory = './data/'
    goodDict = {}
    for filename in os.listdir(directory):
        if "Class" in filename:
            classList = np.loadtxt(directory + filename)
            if not 7 in classList:
                if not 6 in classList:
                    name = filename.split("_Class")[0].split("_0_")[0]
                    goodDict[name] = ''
    dict = {}
    for filename in os.listdir(directory):
        name = filename.split("_Class")[0].split("_0_")[0]
        if name in goodDict.keys():
            if name in dict.keys():
                if 'image' in filename:
                    dict[name][0] = np.loadtxt(directory + filename)
                elif 'Class' in filename:
                    dict[name][1] = np.loadtxt(directory + filename)
            else:
                dict[name] = [-1,-1]
                if 'image' in filename:
                    dict[name][0] = np.loadtxt(directory + filename)
                elif 'Class' in filename:
                    dict[name][1] = np.loadtxt(directory + filename)

    images = []
    labels = []
    numClass = config.label_dim
    filenames = []
    for key in dict.keys():
        if type(dict[key][0]) != int and type(dict[key][1]) != int:
            images.append(dict[key][0])
            label = np.zeros((2000,numClass))
            temp_label = dict[key][1].copy()
            filenames.append(key)
            for i in range(2000):
                label[i,int(temp_label[0,i])] = 1
            labels.append(label) 
    
    train_lst = np.load('cvData/' + config.data + '/splits/train_lst_' + str(study_num) + '.npy')
    val_lst = np.load('cvData/' + config.data + '/splits/val_lst_' + str(study_num) + '.npy')
    x_train = []
    y_train = []
    x_test = []
    y_test = []
    for i in range(len(filenames)):
        filename = filenames[i]
        study = filename
        if study in train_lst:
            x_train.append(images[i])
            y_train.append(labels[i])
        else:
            x_test.append(images[i])
            y_test.append(labels[i])
                    
    x_train = np.array(x_train).reshape((-1,12,2000,1))
    x_test = np.array(x_test).reshape((-1,12,2000,1))
    y_train = np.array(y_train).reshape((-1,1,2000,6))
    y_test = np.array(y_test).reshape((-1,1,2000,6))
    return x_train, x_test, y_train, y_test

modelDir = "/media/deoraid03/rdeo/ecg/models/unet_final/filter_19_dropout/train/0/"

with open('config.json', 'r+') as f:
    config = edict(json.load(f))

#tf.reset_default_graph()
tf.compat.v1.reset_default_graph()
sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(allow_soft_placement=True))
model = load_model(config, sess)

# create saver and load in data 

saver = tf.train.Saver(tf.global_variables(), max_to_keep=None)

# initialize model
ckpt = tf.train.get_checkpoint_state(modelDir)
if ckpt:
    saver.restore(sess, ckpt.model_checkpoint_path)
else:
    sess.run(tf.global_variables_initializer())  

x_train, x_test, y_train, y_test = load_data(config, '0')

ecgDict = {}
ecgDict[0] =   0 # p
ecgDict[1] =  0 # pr
ecgDict[2] = 0 # qrs
ecgDict[3] =  0 # st
ecgDict[4] =   0 # t
ecgDict[5] =  0 # tp
totalCount = 0

for i in range(len(y_train)):
    for j in range(len(y_train[i,0])):
        k = np.argmax(y_train[i,0,j],0)
        ecgDict[k] = ecgDict[k] + 1
        totalCount = totalCount + 1
        
for ecgClass in ecgDict.keys():
    print(ecgClass, ecgDict[ecgClass]/totalCount)        
    
startproblist = [ecgDict[0]/totalCount, 
                 ecgDict[1]/totalCount,
                 ecgDict[2]/totalCount,
                 ecgDict[3]/totalCount,
                 ecgDict[4]/totalCount,
                 ecgDict[5]/totalCount,
                ]

numClass = 6
classList = range(numClass)

ecgNumDict = {}
for i in classList:
    ecgNumDict[str(i)] = {}
    for j in classList:
        ecgNumDict[str(i)][str(j)] = 0
        
for i in range(x_test.shape[0]):
    gt = np.argmax(y_test[i,:,:,:], 2)
    pred = np.argmax(model.predict(x_test[i:i+1])[0,:,:,:], 2)
    for j in range(len(gt[0])):
        ecgNumDict[str(gt[0][j])][str(pred[0][j])] = ecgNumDict[str(gt[0][j])][str(pred[0][j])] + 1

hmm_model = hmm.MultinomialHMM(n_components=numClass)
hmm_model.startprob_ = np.array(startproblist)
hmm_model.transmat_ = np.array([[0.99, 0.01, 0.0, 0.0, 0.0, 0.0], #P
                            [0.00000005, 0.9759999005, 0.024, 0.0, 0.0, 0.0], #PR
                            [0.0, 0.0, 0.99, 0.01, 0.0, 0.0], #QRS
                            [0.0, 0.0, 0.0, 0.988, 0.012, 0.0], #ST
                            [0.0, 0.0, 0.0, 0.0, 0.99, 0.01], #T
                            [0.000000049995, 0.0, 0.000000000005, 0.0, 0.0, 0.9999999] #TP
                               ])

hmm_model.emissionprob_ = np.empty([numClass,numClass])
for gt in classList:
    total = np.sum(ecgNumDict[str(gt)].values())
    for pred in classList:
        hmm_model.emissionprob_[(pred, gt)] =  np.around(ecgNumDict[str(gt)][str(pred)]/total, 3)

filename = "9ef71d292e7a5ea8b3395f7746e7d7e6c63bc44307d4e81ebac11e0e09a0_input.csv"

ecgClassList = ["P", "PR", "QRS", "QT", "RR"]
ecgVecClassList = ["P-PR", "QRS", "ST", "TP"]
intervalDict = {} 
intervalVectorDict = {} 
finalVectorDict = {} 
vectorLabelList = []
for ecgClass in ecgClassList: 
    intervalDict[ecgClass] = [] 
for ecgClass in ecgVecClassList: 
    intervalVectorDict[ecgClass] = {} 
    for lead in range(12): 
        intervalVectorDict[ecgClass][lead] = [] 

csv_file = np.loadtxt(filename, skiprows=1, delimiter=',', usecols=range(12)) 
image = csv_file.T 
image = image - np.median(image,1).reshape((12,1)) 
imageUse = np.apply_along_axis(expandLead, 1, image) 
for j in range(0, 8500, 500):  
    pred = np.argmax(model.predict(smoothRows(imageUse.reshape((1,12,10000,1))[:,:,j:j+2000,:]))[0,:,:,:], 2)[0]
    X = np.atleast_2d(pred).T
    hmmOut = hmm_model.decode(X)[1]
    hmmOut = bridgeAcrossFinal(bridgeUp(bridgeAcross(hmmOut, 0.6), 0.9), 0.9)
    hmmContiguousClasses = rle(hmmOut)
    for k in range(1, len(hmmContiguousClasses[0])-3):
        output = hmmContiguousClasses[0][k]
        outputNext = hmmContiguousClasses[0][k+1]
        outputNextNext = hmmContiguousClasses[0][k+2]
        start = hmmContiguousClasses[1][k]
        end = hmmContiguousClasses[1][k+1]
        endNext = hmmContiguousClasses[1][k+2]
        ecgClass = hmmContiguousClasses[2][k]
        ecgNextClass = hmmContiguousClasses[2][k + 1]
        ecgNextNextClass = hmmContiguousClasses[2][k + 2]
        if not 1 in hmmContiguousClasses[0][k-1:k+3]:
            if ecgClass == 0:
                if ecgNextNextClass == 2:
                    intervalDict["P"].append(output)
                    intervalDict["PR"].append(output + outputNext)
                    for lead in range(12):
                        intervalVectorDict["P-PR"][lead].append(resizeLead(imageUse[lead,(j+start):(j+endNext)], 20))
            elif ecgClass == 2:
                intervalDict["QRS"].append(output)
                intervalDict["QT"].append(output + outputNext + outputNextNext)
                for lead in range(12):
                    intervalVectorDict["QRS"][lead].append(resizeLead(imageUse[lead,(j+start):(j+end)], 20))
            elif ecgClass == 3:
                for lead in range(12):
                    intervalVectorDict["ST"][lead].append(resizeLead(imageUse[lead,(j+start):(j+endNext)], 20))
            #elif ecgClass == 5:
            #    for lead in range(12):
            #        intervalVectorDict["TP"][lead].append(resizeLead(imageUse[lead,(j+start):(j+endNext)], 10))
    for k in range(1, len(hmmContiguousClasses[0])-6):
        output = hmmContiguousClasses[0][k]
        outputNextBeat = hmmContiguousClasses[0][k+6]
        start = hmmContiguousClasses[1][k]
        end = hmmContiguousClasses[1][k+6]
        if not 1 in hmmContiguousClasses[0][k-1:k+6]:
            if ecgClass == 2:
                outputAll = 0
                for choice in range(6):
                    outputAll = outputAll + hmmContiguousClasses[0][k + choice]
                intervalDict["RR"].append(outputAll)

for ecgClass in intervalDict.keys():
    newKey = ecgClass + "dur"
    vectorLabelList.append(newKey)
    finalVectorDict[newKey] = np.nanmedian(intervalDict[ecgClass])

for ecgClass in intervalVectorDict.keys():
    for lead in intervalVectorDict[ecgClass]:
        newKey = ecgClass + str(lead) + "vec"
        vectorLabelList.append(newKey)
        if not intervalVectorDict[ecgClass][lead] == []:
            finalVectorDict[newKey] = np.nanmedian(intervalVectorDict[ecgClass][lead], axis = 0)
        else:
            finalVectorDict[newKey] = []

RRpred = 60000/finalVectorDict["RRdur"]
PRpred = finalVectorDict["PRdur"]
QRSpred = finalVectorDict["QRSdur"]        
QTpred = finalVectorDict["QTdur"] 
print("Heart rate", RRpred, "PR", PRpred, "QRS", QRSpred, "QT", QTpred)

out = open(filename.split("_input")[0]  + "_voltageVector.txt", 'w')
for j in vectorLabelList:
    if not "dur" in j:
        if not finalVectorDict[j] == []:
            for k in finalVectorDict[j]:
                out.write(str(k) + "\n")
        else:
            out.write("NA\n")
    else:
        out.write(str(finalVectorDict[j]) + "\n")
out.close()
