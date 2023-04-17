
import tensorflow as tf 
import numpy as np
import math
import os
		
from datetime import datetime

LAYER1_SIZE = 128
LAYER2_SIZE = 64

class CriticNetwork:
	"""docstring for CriticNetwork"""
	def __init__(self, sess, state_dim, action_dim, lr, tau, l2):
		self.lr = lr;
		self.tau = tau;
		self.l2 = l2;
		
		self.time_step = 0
		self.sess = sess
		# create q network
		self.state_input,\
		self.action_input,\
		self.q_value_output,\
		self.net = self.create_q_network(state_dim,action_dim)
		
		self.losses=[]
		self.total_loss = 0
		self.is_episode_done = False

		# create target q network (the same structure with q network)
		self.target_state_input,\
		self.target_action_input,\
		self.target_q_value_output,\
		self.target_update = self.create_target_q_network(state_dim,action_dim,self.net)

		self.create_training_method()

		# initialization 
		self.sess.run(tf.initialize_all_variables())
			
		self.update_target()

	def create_training_method(self):
		# Define training optimizer
		self.y_input = tf.placeholder("float",[None,1])
		weight_decay = tf.add_n([self.l2 * tf.nn.l2_loss(var) for var in self.net])
		self.cost = tf.reduce_mean(tf.square(self.y_input - self.q_value_output)) + weight_decay
		self.optimizer = tf.train.AdamOptimizer(self.lr).minimize(self.cost)
		self.action_gradients = tf.gradients(self.q_value_output,self.action_input)

	def create_q_network(self,state_dim,action_dim):
		# the layer size could be changed
		layer1_size = LAYER1_SIZE
		layer2_size = LAYER2_SIZE

		state_input = tf.placeholder("float",[None,state_dim])
		action_input = tf.placeholder("float",[None,action_dim])

		W1 = self.variable([state_dim,layer1_size],state_dim)
		b1 = self.variable([layer1_size],state_dim)
		W2 = self.variable([layer1_size,layer2_size],layer1_size+action_dim)
		W2_action = self.variable([action_dim,layer2_size],layer1_size+action_dim)
		b2 = self.variable([layer2_size],layer1_size+action_dim)
		W3 = tf.Variable(tf.random_uniform([layer2_size,1],-3e-3,3e-3))
		b3 = tf.Variable(tf.random_uniform([1],-3e-3,3e-3))

		layer1 = tf.nn.relu(tf.matmul(state_input,W1) + b1)
		layer2 = tf.nn.relu(tf.matmul(layer1,W2) + tf.matmul(action_input,W2_action) + b2)
		q_value_output = tf.identity(tf.matmul(layer2,W3) + b3)

		return state_input,action_input,q_value_output,[W1,b1,W2,W2_action,b2,W3,b3]

	def create_target_q_network(self,state_dim,action_dim,net):
		state_input = tf.placeholder("float",[None,state_dim])
		action_input = tf.placeholder("float",[None,action_dim])

		ema = tf.train.ExponentialMovingAverage(decay=1-self.tau)
		target_update = ema.apply(net)
		target_net = [ema.average(x) for x in net]

		layer1 = tf.nn.relu(tf.matmul(state_input,target_net[0]) + target_net[1])
		layer2 = tf.nn.relu(tf.matmul(layer1,target_net[2]) + tf.matmul(action_input,target_net[3]) + target_net[4])
		q_value_output = tf.identity(tf.matmul(layer2,target_net[5]) + target_net[6])

		return state_input,action_input,q_value_output,target_update

	def update_target(self):
		self.sess.run(self.target_update)

	def train(self,y_batch,state_batch,action_batch):
		if self.is_episode_done and self.time_step > 0:
			self.losses.append(self.total_loss / self.time_step)
			self.total_loss = 0
			self.is_episode_done = False

		self.time_step += 1
		loss,_= self.sess.run([self.cost,self.optimizer],feed_dict={
			self.y_input:y_batch,
			self.state_input:state_batch,
			self.action_input:action_batch
			})
		self.total_loss += loss

	def gradients(self,state_batch,action_batch):
		return self.sess.run(self.action_gradients,feed_dict={
			self.state_input:state_batch,
			self.action_input:action_batch
			})[0]

	def target_q(self,state_batch,action_batch):
		return self.sess.run(self.target_q_value_output,feed_dict={
			self.target_state_input:state_batch,
			self.target_action_input:action_batch
			})

	def q_value(self,state_batch,action_batch):
		return self.sess.run(self.q_value_output,feed_dict={
			self.state_input:state_batch,
			self.action_input:action_batch})

	# f fan-in size
	def variable(self,shape,f):
		return tf.Variable(tf.random_uniform(shape,-1/math.sqrt(f),1/math.sqrt(f)))

	def load_network(self, model_name):
		checkpoint = tf.train.get_checkpoint_state("saved_critic_networks/" + model_name)
		if checkpoint and checkpoint.model_checkpoint_path:
			tf.train.Saver().restore(self.sess, checkpoint.model_checkpoint_path)
			print("Successfully loaded:", checkpoint.model_checkpoint_path)
		else:
			print("Could not find old network weights")

	def save_network(self,time_step, model_name):
		now = datetime.now()
		save_path = 'saved_critic_networks/' + now.strftime("%d_%m_%Y") + "/"
		if not os.path.exists(save_path):
			os.mkdir(save_path)
		save_path = save_path + "/" + model_name
		if not os.path.exists(save_path):
			os.mkdir(save_path)
			

		save_path = 'saved_critic_networks/' + now.strftime("%d_%m_%Y") + "/" + model_name + "/" + now.strftime("%H_%M_%S")
		tf.train.Saver().save(self.sess, save_path, global_step = time_step)
		print('Critic network saved...   ' + save_path,time_step)

		