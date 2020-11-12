# CardioNexus & TickerCardiology - echocardiogram view prediction

Author: k.mcnamara
Date: 12/11/2020

Tensorflow: 2.1.0
Python: 3.6


## Original EchoCV 
Code: ./echocv/
Original EchoCV models: ./echocv_models/

## To train model:
See train_vgg.ipynb.
Training, validation data flows from S3
Code: ./model_training/

## To use model:
See deploy_predict_vgg.ipynb
Download patient from S3 and store in this notebook instance to local directory ./inputs/
Upload patient results to S3 from local directory ./outputs/

## Supplementary:
Code: ./support_funcs/

### Note:
./other/ - contains notebooks for targz, git, confusion matrix, reading edf files etc
./ecgai/ - code for Rahul's EcgAI 
