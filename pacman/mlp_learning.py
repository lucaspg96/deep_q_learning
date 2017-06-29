from __future__ import print_function

import skimage as skimage
from skimage import transform, color, exposure
from skimage.transform import rotate
from skimage.viewer import ImageViewer
from os import system as sys
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

ACTIONS = 0 # number of valid actions
GAMMA = 0.95 # decay rate of past observations
REPLAY_MEMORY = 50000 # number of previous transitions to remember
BATCH = 0 # size of minibatch
LEARNING_RATE = 0.05
FRAMES = 1
epsilon = 1
epsilon_decay = 0.99
model_shape = None
step = 0
game = ""

def buildmodel(name):
    print("Initialyzing model...")
    model = Sequential()
    print("Model dim:",model_shape)
    model.add(Dense(18,activation="sigmoid",input_dim=model_shape))
    model.add(Dense(ACTIONS,activation="sigmoid"))
    # model.add(Dense(20,activation="sigmoid"))
    # model.add(Dense(ACTIONS))
       
    #optimizer = Adam(lr=LEARNING_RATE, decay = 0.001)
    optimizer = SGD(lr=LEARNING_RATE,decay=0.01)
    model.compile(loss='mse',optimizer=optimizer,metrics=["accuracy"])
    print("We finish building the model")

    try:
        print("Trying load model weights for {}".format(name))
        # with open("{}.json".format(name)) as f:
        #     model = model_from_json(f.read())
        model = load_model("{}/model.h5".format(name))
        print("Weights loaded")

    except Exception as e:
        #print(e)
        print("Model weights not found".format(name))

        # #model.save_weights("{}.h5".format(name), overwrite=True)
        # with open("{}.json".format(name), "w") as outfile:
        #     json.dump(model.to_json(), outfile)
    
    return model

model = None
last_state = None
D = None

def init(n_actions,game_name,observation,batch=32):
    global ACTIONS
    global model
    global step
    global game
    global model_shape
    global BATCH
    global D
    global last_state


    step = 0
    ACTIONS = n_actions
    BATCH = batch
    D = deque()
    model_shape = len(observation)*FRAMES
    observation = np.array(list(observation)*FRAMES)
    last_state = np.reshape(observation,(1,model_shape))

    game = game_name    

    try:
        sys("mkdir {}".format(game))
    except:
        pass

    model = buildmodel(game)

def getAction(data,randomAction=False):
    global step
    global OBSERVE
    global model
    global last_state
    global epsilon


    step += 1
    last_state[0,(model_shape-int(model_shape/FRAMES)):model_shape] = data
    
    #case of observation
    if randomAction:
        return random.choice(range(ACTIONS))

    #asking to the agent
    else:
        #epsilon-gredy
        if(random.random()<=epsilon):
            #print("********************* \n*   RANDOM ACTION   *\n*********************")
            action_index = random.choice(range(ACTIONS))
        
        #predict
        else:
            q = model.predict(last_state)
            max_Q = np.argmax(q)
            action_index = max_Q
        
        #decreasing epsilon 
        epsilon *= epsilon_decay

        return action_index

def storeMem(data,r_t,action):
    global D
    global last_state
    state = last_state
    #print(state.shape,model_shape-int(model_shape/FRAMES),model_shape)
    state[0,(model_shape-int(model_shape/FRAMES)):model_shape] = data

    #state = np.reshape(data,(1,model_shape))
    D.append((last_state, action, r_t, state))

    last_state = state

    if len(D) > REPLAY_MEMORY:
        D.popleft()

def train(data,r_t,action):    
    #run the selected action and observed next state and reward
    storeMem(data,r_t,action)

    global D
    global model
    loss = 0
    #sample a minibatch to train on
    if(len(D)<BATCH):
        # n = len(D)
        # minibatch = random.sample(D, n)
        # inputs = np.zeros((n, model_shape)) 
        # targets = np.zeros((n, ACTIONS))
        minibatch  = random.sample(D,1)
        inputs = np.zeros((1,model_shape))
        targets = np.zeros((1,ACTIONS))

    else:
        minibatch = random.sample(D, BATCH)
        inputs = np.zeros((BATCH, model_shape)) 
        targets = np.zeros((BATCH, ACTIONS))

    #Now we do the experience replay
    for i in range(0, len(minibatch)):
        state_t = minibatch[i][0]
        action_t = minibatch[i][1]   #This is action index
        reward_t = minibatch[i][2]
        state_t1 = minibatch[i][3]
        # if terminated, only equals reward

        inputs[i:i + 1] = state_t    #I saved down s_t
        #print(state_t.shape,inputs.shape,targets.shape,model_shape)
        a = model.predict(state_t)
        #print(a)
        targets[i] = a  # Hitting each buttom probability
        Q_sa = model.predict(state_t1)

        targets[i, action_t] = reward_t + GAMMA * np.max(Q_sa)

    loss += sum(model.train_on_batch(inputs, targets))
    #print(loss)
    # # save progress every 100 iterations
    # if step % 300 == 0:
    #     global game
    #     print("Saving the model")
    #     model.save("{}.h5".format(game), overwrite=True)
    #     with open("{}.json".format(game), "w") as outfile:
    #         json.dump(model.to_json(), outfile)
    return loss

def saveModel():
    model.save("{}/model.h5".format(game), overwrite=True)

def saveStatistics(statistics):
    for s in statistics:
        with open("{}/{}.csv".format(game,s),"w") as f:
            f.write("\n".join(map(str,statistics[s])))