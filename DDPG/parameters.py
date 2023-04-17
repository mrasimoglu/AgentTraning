from __future__ import print_function
from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets

_lr = 0.0005
_tau = 0.005
_l2 = 0.01
_episodes = 200000

def f_lr(learning_rate):
    try:
        gym
    except NameError:
        a = 0
    else:
        _lr = learning_rate
        gym.actor.actor_network.lr = learning_rate
        gym.actor.critic_network.lr = learning_rate
    return learning_rate
    
def f_tau(tau):
    try:
        gym
    except NameError:
        a = 0
    else:
        _tau = tau
        gym.actor.actor_network.lr = tau
        gym.actor.critic_network.lr = tau
    return tau
    
def f_l2(l2):
    try:
        gym
    except NameError:
        a = 0
    else:
        _l2 = l2
        gym.actor.actor_network.lr = l2
        gym.actor.critic_network.lr = l2
    return l2
    
def f_episodes(episodes):
    try:
        gym
    except NameError:
        a = 0
    else:
        _episodes = episodes
        gym.episodes = episodes
        gym.episodes = episodes
    return episodes

interact(f_episodes, episodes=widgets.IntSlider(min=0, max=100000, description='Episode Count', step=1, value=_episodes));
interact(f_lr, learning_rate=widgets.FloatSlider(min=0, max=1, description='Learning Rate', step=0.0001, value=_lr));
interact(f_tau, tau=widgets.FloatSlider(min=0, max=1, description='Tau', step=0.0001, value=_tau));
interact(f_l2, l2=widgets.FloatSlider(min=0, max=1, description='L2', step=0.0001, value=_l2));