
import Image
import numpy as np
import numpy.random as npr
import scipy.misc as spm

def next_batch(i,path,num_channels):
    length_of_i = i.size
    xtrain = np.empty([length_of_i,224*224*num_channels],dtype=np.float32)
    ytrain = np.empty([length_of_i,5],dtype=np.float32)
    counter = np.int(0)
    for sample in i:
        img = Image.open(path + "/" + str(sample) + ".jpg")
        if img.mode == 'CMYK':
            img = img.convert('RGB')
        img = img.resize((224,224))
        img = np.array(img,dtype=np.float32)
        img = np.concatenate(np.concatenate(img))
        xtrain[counter,:] = img/255
        if sample < 100:
            ytrain[counter,:] = np.array([1.0, 0.0, 0.0, 0.0, 0.0])
        elif 100 <= sample < 272:
            ytrain[counter,:] = np.array([0.0, 1.0, 0.0, 0.0, 0.0])
        elif 272 <= sample < 365:
            ytrain[counter,:] = np.array([0.0, 0.0, 1.0, 0.0, 0.0])
        elif 365 <= sample < 523:
            ytrain[counter,:] = np.array([0.0, 0.0, 0.0, 1.0, 0.0])
        elif 523 <= sample < 597:
            ytrain[counter,:] = np.array([0.0, 0.0, 0.0, 0.0, 1.0])
        #elif 24999 < sample <= 29999:
            #ytrain[counter,:] = np.array([0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0])
        #elif 29999 < sample <= 34999:
            #ytrain[counter,:] = np.array([0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0])
        #elif 34999 < sample <= 39999:
            #ytrain[counter,:] = np.array([0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0])
        counter += 1
    return xtrain, ytrain
