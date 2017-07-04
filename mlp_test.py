import gym
import mlp_learning as learn
from tqdm import trange
import os

def run_game(game_name,epochs,ev,observe=100):
	env = gym.make(game_name)
	observation = env.reset() # reset for each new trial
	learn.init(env.action_space.n,game_name,observation,batch=32)
	step = 0
	points = []

	os.system("clear")
	print("Starting observations:")
	for _ in trange(observe):
		action = learn.getAction(observation,randomAction=True)
		observation, reward, done, info = env.step(action)
		learn.train(observation,reward,action)
		step+=1
		if done:
			env.reset()

	losses = []
	scores = []
	record = 0

	print("Starting train:")
	for _ in trange(epochs):
		#keep restarting game when ends 
		#t=0 # steps on match
		s=0 # Scores

		while True:
			#env.render()
			#t+=1
			step+=1 #global step

			action = learn.getAction(observation)

			observation, reward, done, info = env.step(action)

			s+=reward

			losses.append(learn.train(observation,reward,action))

			if done:
				scores.append(s)
				break
			
			if step%1000==0:
				learn.saveModel()

		observation = env.reset() # reset for each new trial

	learn.saveModel()
	print("Evaluating:")
	observation = env.reset()
	for i in trange(ev):
		 t = 0
		 s=0
		 while True: 
			 #env.render()
			 t+=1
			
			 action = learn.getAction(observation,evaluate=True)
			 observation, reward, done, info = env.step(action)
			 s+=reward
			 #learn.storeMem(observation,reward,action)
			 if done:
			 	points.append(s)
			 	break

		 observation = env.reset() # reset for each new trial

	statistics = {
		"train_loss":losses,
		"train_scores": scores,
		"test_scores": points
	}
	learn.saveStatistics(statistics)


#Main bloc ---------------------------------------
games = [ \
		'CartPole-v0' \
		#'MsPacman-ram-v0'
		# ,'MountainCar-v0' \
		#'LunarLander-v2' \
		# 'MsPacman-v0'
		#'SpaceInvaders-v0'
		#'Pong-v0'
		#'Breakout-v0'
		]
for game_name in games:
	epochs = 10000
	ev = 1000
	run_game(game_name,epochs,ev,1000)