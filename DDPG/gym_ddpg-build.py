from mlagents_envs.environment import UnityEnvironment
from ddpg import *
import os
import matplotlib.pyplot as plt
from IPython.display import clear_output
import tensorflow as tf
import datetime
import matplotlib.pyplot as plt
import numpy as np
class GymDDPG:
    def __init__(self, parameters):
        self.episode = 0
        self.episodes = parameters._episodes

        print("Waiting Unity Editor")
        self.env = UnityEnvironment("build/Drone.exe")
        self.env.step()

        print("Connected to Unity")
        self.agents = []
        self.model_name = ""

        behaviour_names = self.env.get_behavior_names()
        for i in range(len(behaviour_names)):
            spec = self.env.get_behavior_spec(behaviour_names[i])
            self.agents.append(DDPG(self.env, spec.observation_shapes[0][0], spec.action_size, parameters._lr, parameters._tau, parameters._l2))

    def set_episode(self, ep):
        self.episode = ep

    def get_episode(self):
        return self.episode

    def train(self):
        rewards2print=[]
        # current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        behaviour_names = self.env.get_behavior_names()
        # train_log_dir = 'logs/gradient_tape/' + current_time + '/train'
        # #test_log_dir = 'logs/gradient_tape/' + current_time + '/test'
        # train_summary_writer = tf.summary.FileWriter(train_log_dir)
        for agent in self.agents:
            print("\nSelect model to load")
            models = os.listdir("saved_actor_networks\\")
            for i in range(len(models)):
                print("\t" + str(i + 1) + "- " + models[i])

            model_to_load = int(input("Model to load (0 for new model): \n"))
            if model_to_load != 0:
                agent.critic_network.load_network(models[model_to_load - 1])
                agent.actor_network.load_network(models[model_to_load - 1])
            
            self.model_name = str(input("New model name: \n"))

        total_rewards = [[]] * len(self.agents)

        while self.episode < self.episodes:
            self.env.reset()
            
            states = []
            for i in range(len(self.agents)):
                state_info, _ = self.env.get_steps(self.env.get_behavior_names()[i])
                states.append(state_info.obs[0].reshape(self.agents[i].state_dim))

            done = False
            
            total_reward = 0
            while not done:
                for i in range(len(self.agents)):
                    action = self.agents[i].noise_action(states[i])

                    self.env.step()
                    self.env.set_actions(behaviour_names[i], action.reshape(1, self.agents[i].action_dim))

                    nextstate_info, terminal_step = self.env.get_steps(behaviour_names[0])
                    next_state = nextstate_info.obs[0].reshape(self.agents[i].state_dim)
                    done = True if terminal_step else False
                    reward = nextstate_info.reward
                    # print("\nCurrent reward: ",reward)
                    total_reward += reward
                    
                    self.agents[i].perceive(states[i],action,reward,next_state,done)
                    states[i] = next_state
            print("Episode reward: ", total_reward,"\n Episode",self.episode)
            rewards2print.append(total_reward)
            # train_summary_writer.scalar(total_reward,self.episode)
                 # tf.summary.scalar('reward', total_reward, step=self.episode)
            
            # clear_output(wait=False)
            for i in range(len(self.agents)):
                if self.episode > 0 and self.episode % 100 == 0:
                    clear_output(wait=False)
                    x_axis=np.arange(self.episode+1)
                    plt.plot(x_axis,rewards2print)
                    plt.ylabel("reward")
                    plt.xlabel("episode")
                    
                    plt.savefig("logs/reward"+str(self.episode)+".png")
                    self.agents[i].actor_network.save_network(self.episode, self.model_name)
                    self.agents[i].critic_network.save_network(self.episode, self.model_name)

                # total_rewards[i].append(total_reward)
                # plt.plot(range(0, self.episode + 1), total_rewards[i])
                # plt.label(behaviour_names[i])
                # plt.xlabel('Episodes')
                # plt.ylabel('Reward')
                # plt.show()

            self.episode += 1

if __name__ == "__main__":
    import parameters

    gym = GymDDPG(parameters)
    gym.train()