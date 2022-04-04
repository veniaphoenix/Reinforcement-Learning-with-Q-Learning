import gym 
import numpy as np 
import random

env = gym.make("MountainCar-v0")
env.reset()
#Create Q - table
q_table_size = [20, 20]

print(env.reset())


#Create segmentation of Q - Table
q_table_segment_size = (env.observation_space.high - env.observation_space.low)/q_table_size

print(q_table_segment_size)

print(env.observation_space.high)
print(env.observation_space.low)

#Change state to integers
def convert_state(real_state):
	real_state = (real_state - env.observation_space.low) // q_table_segment_size
	return tuple(real_state.astype(np.int))

q_table = np.random.uniform(
	low = -2, 
	high = 0, 
	size = (q_table_size + [env.action_space.n])
	)
#Load pretrained_q_table
with open('pretrained_q_table', 'rb') as f:
	pretrained_q_table = np.load(f)

q_table = pretrained_q_table

print(q_table.shape)

learning_rate = 0.1
discount_factor = 0.9
eps = 1000 
show_agent = 250
max_reward = -999
epsilon = 0.9
ep_start_epsilon = 0
ep_end_epsilon = (eps // 2)
epsilon_decay_each_ep = epsilon / (ep_end_epsilon - ep_start_epsilon)
print(epsilon_decay_each_ep)
action_in_final_episodes = [] #Use to display the action list in the final episode
reach_goal_counter = 0 
for ep in range(eps):
	
	print("Episode: {}".format(ep))
	done = False
	current_state = convert_state(env.reset())
	all_action = []
	ep_reward = 0 
	start_state = current_state
	

	while not done:

		random = np.random.random()

		if random > epsilon:
			action = np.argmax(q_table[current_state])
			
		else: 
			action = np.random.randint(0,env.action_space.n)
		all_action.append(action)

		#if ep % 100 == 0:
			#env.render()

		

		#Do action
		new_real_state, reward, done, _ = env.step(action = action)
		ep_reward += reward

		if done:
			#Check the x is larger than the flag
			if new_real_state[0] >= env.goal_position:
				print("You reached goal!!!") 
				reach_goal_counter += 1
				if ep_reward > max_reward:
					ep_max_reward = ep
					max_reward = ep_reward
					action_in_best_episode = all_action
					final_eps_state = start_state
					with open('pretrained_q_table','wb') as f:
						np.save(f, q_table)
					

		#Convert new state
		new_state = convert_state(new_real_state)

		#Update Q - value of the of the previous state
		current_q_value = q_table[current_state + (action,)]

		new_q_value = (1-learning_rate)*current_q_value + learning_rate*(reward + discount_factor*(np.max(q_table[new_state])))

		q_table[current_state + (action,)] = new_q_value

		current_state = new_state
	
		if ep_end_epsilon >= ep >= ep_start_epsilon:

			epsilon = epsilon - epsilon_decay_each_ep
accuracy = (reach_goal_counter / eps) *100

print("max reward: {} in ep {},\naction in best episode: {}\nACCURACY = {}%".format(max_reward, ep_max_reward, np.array(action_in_best_episode), accuracy))

#Test action in best episode
env.reset()
env.state = convert_state(final_eps_state)
for action in action_in_best_episode:
	env.step(action)
	env.render()








