from __future__ import print_function

import skimage as skimage
from skimage import transform, color, exposure
from skimage.transform import rotate
from skimage.viewer import ImageViewer
import sys
import os
import random
import numpy as np
from collections import deque
import json
from keras import initializers
from keras.initializers import normal, identity
from keras.models import model_from_json, load_model
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD , Adam
import tensorflow as tf


from skimage import io, exposure, img_as_uint, img_as_float

ACTIONS = 4 # number of valid actions
GAMMA = 0.99 # decay rate of past observations
REPLAY_MEMORY = 50000 # number of previous transitions to remember
BATCH = 32 # size of minibatch
LEARNING_RATE = 0.005
epsilon = 1
epsilon_decay = 0.99

img_rows , img_cols = 80, 80
#Convert image into Black and white
img_channels = 4 #We stack 4 frames
model_shape = None
step = 1
game = ""

def buildmodel(name):
    global model_shape

    model = Sequential()
  
    model.add(Convolution2D(32, 8, 8,activation="relu", subsample=(4, 4), border_mode='same',input_shape=(model_shape)))  #80*80*4
    # model.add(Activation('relu'))
    model.add(Convolution2D(64, 5, 5,activation="relu", subsample=(2, 2), border_mode='same'))
    # model.add(Activation('relu'))
    model.add(Convolution2D(64, 3, 3,activation="relu", subsample=(1, 1), border_mode='same'))
    # model.add(Activation('relu'))
    
    model.add(Flatten())
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dense(ACTIONS, activation="sigmoid"))
       
    adam = Adam(lr=LEARNING_RATE, decay = 0.001)
    model.compile(loss='mse',optimizer=adam)
    print("We finish building the model")

    try:
        print("Trying load model weights for {}".format(name))
        # with open("{}.json".format(name)) as f:
        #     model = model_from_json(f.read())
        model.load_wheigts("{}/model.h5".format(name))
        print("Weights loaded")

    except Exception as e:
        print("Model weights not found")

        # #model.save_weights("{}.h5".format(name), overwrite=True)
        # with open("{}.json".format(name), "w") as outfile:
        #     json.dump(model.to_json(), outfile)
    
    return model

model = None

D = None
s_t = []

def processImage(image):
    if len(image.shape)==1:
        x_t = np.reshape(image,(image.shape[0],1))
    else:
        x_t = skimage.color.rgb2gray(image)
        x_t = skimage.transform.resize(x_t,(img_rows,img_cols))
        x_t = skimage.exposure.rescale_intensity(x_t,out_range=(0,255))
    return x_t

def init(n_actions,game_name,image,batch=100):
    global s_t
    global ACTIONS
    global model
    global step
    global game
    global model_shape
    global BATCH
    global D

    os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
    
    step = 0
    ACTIONS = n_actions
    BATCH = batch
    D = deque()
    x_t = processImage(image)

    if len(image.shape)==1:
        model_shape = (image.shape[0],1,img_channels)
        game = game_name    
        
    else:
        model_shape = (img_rows,img_cols,4)
        game = "{}({}x{})".format(game_name,x_t.shape[0],x_t.shape[1])
        saveImage(x_t)
    
    model = buildmodel(game)
    
    s_t = np.stack((x_t,x_t,x_t,x_t),axis=2)
    s_t = s_t.reshape(1, s_t.shape[0], s_t.shape[1], s_t.shape[2])  #1*80*80*4 
    print(s_t.shape,model_shape)  

def getAction(image,randomAction=False,evaluate=False):
    global s_t
    global step
    global OBSERVE
    global model
    global epsilon

    step += 1

    # if(len(D)<=BATCH):
    #     return random.choice(range(ACTIONS))
    # else:
    x_t = processImage(image)

    x_t = np.reshape(x_t,(1,x_t.shape[0],x_t.shape[1],1))
    
    s_t = np.append(x_t,s_t[:, :, :, :3], axis=3)
    
    
    #case of observation
    if randomAction:
        return random.choice(range(ACTIONS))

    #asking to the agent
    else:
        # on evaluate, always get the predicted action
        if evaluate:
            q = model.predict(s_t)
            max_Q = np.argmax(q)
            action_index = max_Q
            return action_index

        else:
            #epsilon-gredy
            if(random.random()<=epsilon):
                #print("********************* \n*   RANDOM ACTION   *\n*********************")
                action_index = random.choice(range(ACTIONS))
            
            #predict
            else:
                q = model.predict(s_t)
                max_Q = np.argmax(q)
                action_index = max_Q
            
            #decreasing epsilon 
            epsilon *= epsilon_decay

            return action_index

def storeMem(image,r_t,action):
    global s_t
    global D
    #print("Store mem")
    x_t1 = processImage(image)
    x_t1 = np.reshape(x_t1,(1,x_t1.shape[0],x_t1.shape[1],1))

    s_t1 = np.append(x_t1, s_t[:, :, :, :3], axis=3)
    # store the transition in D
    #print(s_t.shape,s_t1.shape)
    D.append((s_t, action, r_t, s_t1))
    s_t = s_t1

    if len(D) > REPLAY_MEMORY:
        D.popleft()

def train(image,r_t,action):
    storeMem(image,r_t,action)

    #run the selected action and observed next state and reward 
    global D
    global model
    loss = 0
    Q_sa = 0
    action_index = 0
    r_t = 0
    s_t,action,r_t,s_t1 = D[-1]
    #sample a minibatch to train on

    if(len(D)<BATCH):
        # n = len(D)
        # minibatch = random.sample(D, n)
        # inputs = np.zeros((n, model_shape)) 
        # targets = np.zeros((n, ACTIONS))
        minibatch  = random.sample(D,1)
        inputs = np.zeros((1,s_t.shape[1], s_t.shape[2], s_t.shape[3]))
        targets = np.zeros((1,ACTIONS))

    else:
        minibatch = random.sample(D, BATCH)
        inputs = np.zeros((BATCH, s_t.shape[1], s_t.shape[2], s_t.shape[3])) 
        targets = np.zeros((BATCH, ACTIONS))

    #Now we do the experience replay
    for i in range(0, len(minibatch)):
        state_t = minibatch[i][0]
        action_t = minibatch[i][1]   #This is action index
        reward_t = minibatch[i][2]
        state_t1 = minibatch[i][3]
        # if terminated, only equals reward

        inputs[i:i + 1] = state_t    #I saved down s_t
        #print(state_t.shape,inputs.shape,targets.shape)
        targets[i] = model.predict(state_t)  # Hitting each buttom probability
        Q_sa = model.predict(state_t1)

        targets[i, action_t] = reward_t + GAMMA * np.max(Q_sa)

    # targets2 = normalize(targets)
    loss += model.train_on_batch(inputs, targets)
    return loss

def saveImage(img):
    global step
    io.use_plugin('freeimage')

    im = exposure.rescale_intensity(img, out_range='float')
    im = img_as_uint(im)

    io.imsave('test({}x{}).png'.format(img.shape[0],img.shape[1]), im)

def saveModel():
    model.save("{}/model.h5".format(game), overwrite=True)

def saveStatistics(statistics):
    for s in statistics:
        with open("{}/{}.csv".format(game,s),"w") as f:
            f.write("\n".join(map(str,statistics[s])))