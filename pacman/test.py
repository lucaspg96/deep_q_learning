import gym
import dqlearn as learn
from tqdm import trange
import bot

def run_game(game_name,epochs,ev,observe=50):
	env = gym.make(game_name)
	observation = env.reset() # reset for each new trial
	learn.init(env.action_space.n,game_name,observation,batch = 10)
	step = 0
	points = []

	print("Starting observations:")
	for i in trange(epochs+ev):
		#keep restarting game when ends 
	    t = 0
	    s=0
	    while True: # run for 100 timesteps or until done, whichever is first
	    	env.render()
	        t+=1
	        if step==observe:
	        	print("Starting train")
	        step+=1
	        
	        if step<=observe:
	        	action = env.action_space.sample()
	        else:
	        	action = learn.getAction(observation)

	        observation, reward, done, info = env.step(action)
	        s+=reward
	        learn.storeMem(observation,reward,action)
	        if step>observe:
	        	loss = learn.train()
	        if done:
	        	if i>epochs:
	        		points.append(s)
	        	break

	    observation = env.reset() # reset for each new trial

	# print("Evaluating:")
	# observation = env.reset()
	# for i in trange(ev):
	#     t = 0
	#     s=0
	#     while True: 
	#         env.render()
	#         t+=1
	        
	#         action = learn.getAction(observation)
	#         observation, reward, done, info = env.step(action)
	#         s+=reward
	#         learn.storeMem(observation,reward,action)
	#         if done:
	#         	points.append(s)
	#         	break

	#     observation = env.reset() # reset for each new trial
	return points

#Main bloc ---------------------------------------
games = ['CartPole-v0', 'MountainCar-v0','LunarLander-v2']
for game_name in games:
	epochs = 10000
	ev = 100
	points = run_game(game_name,epochs,ev,100)
	mean = float(sum(points)/len(points))
	print("Mean score: {}".format(mean))
	bot.send("{} trained {} epochs and evaluated in {} scores".format(game_name,epochs,mean))