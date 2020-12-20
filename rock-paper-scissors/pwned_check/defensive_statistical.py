
import numpy as np
from scipy.stats import beta
import random

def was_i_pwned(n, w_me, w_opp, samples=10000, seed=None):
    """
    n: number of steps played
    w_me: number of my winnign actions
    w_opp: number of opponents winning actions
    samples: number of samples to calculate the probability. Set high for more accuracy, low for better performance
    seed: Set if you want reproducible numebrs
    
    returns the probabiliy of your winning rate being smaller then your opponents winnign rate.
    """
    
    beta_me = beta(1 + w_me, 1 + (n - w_me))  #posterior probability distribution of your win rate
    beta_opp = beta(1 + w_opp, 1 + (n - w_opp))  #posterior probability distribution of opponent win rate
    
    if seed:
        import numpy as np
        np.random.seed(seed=seed)
    
    # calculate probability that opponents win rate is higher than yours from some smaples
    # (did not find a closed solution to do this)
    s_me = beta_me.rvs(samples)
    s_opp = beta_opp.rvs(samples)
    
    return (s_me < s_opp).sum() / samples

def defensive_statistical(observation, configuration):
    global action_histogram
    
    global last_my_action
    global w_me
    global w_opp
    global pwned

    if observation.step == 0:
        action_histogram = {}
        last_my_action = 0
        w_me = 0
        w_opp = 0
        pwned=False
        return
    
    action = observation.lastOpponentAction
    if action not in action_histogram:
        action_histogram[action] = 0

    action_histogram[action] += 1
    mode_action = None
    mode_action_count = None
    for k, v in action_histogram.items():
        if mode_action_count is None or v > mode_action_count:
            mode_action = k
            mode_action_count = v
            continue
    
    my_action = (mode_action + 1) % configuration.signs
    
    if action == (last_my_action + 1) % configuration.signs:
        w_opp += 1
        
    elif last_my_action == (action + 1) % configuration.signs:
        w_me += 1
        
    else:
        pass
    
    last_my_action = my_action
    
    
    prob_pwned = was_i_pwned(observation.step, w_me=w_me, w_opp=w_opp, samples=10000, seed=42)
    
    if observation.step >= 15 and prob_pwned >= 0.98:
        pwned = True
    
    if pwned:
        my_action = random.randint(0, configuration.signs-1)

    return my_action
