
import random
import numpy as np

from scipy.stats import beta
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier

sample_size = 10
opponent_observations = []
my_actions = []
w_me = 0
w_opp = 0

def make_train_set(opponent_observations, my_actions, sample_size):
    observed = np.array([list(x) for x in zip(opponent_observations, my_actions)])
    prediction_sample = observed[-sample_size:].reshape(1, sample_size * 2)

    splits = TimeSeriesSplit(
        max_train_size=sample_size,
        n_splits=len(observed) - sample_size
    )
        
    X = np.array([
            observed[train_index] for train_index, _ in splits.split(observed)
    ])
    
    X = X.reshape(X.shape[0], X.shape[1] * X.shape[2])

    Y = np.array([
            observed[y_index][0][0] for _, y_index in splits.split(observed)
    ])

    Y = Y.reshape(Y.shape[0], 1)
    
    return X, Y, prediction_sample

def action_decision_tree_dev(opponent_observations, my_actions,
    sample_size, configuration=None):

    X, Y, prediction_sample = make_train_set(opponent_observations, my_actions, sample_size)

    Y = np.ravel(Y)
    
    model = RandomForestClassifier(
        n_estimators=100,
        criterion="entropy",
        max_depth = 8,
    )

    model.fit(X, Y)

    prediction = model.predict(prediction_sample)
    
    # action is what beats the opponents prediciton
    action = (prediction + 1) % (configuration.signs if configuration else 3)
    action = int(action[0])
    
    return action

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

def defensive_forest_agent(observation, configuration):
    global opponent_observations
    global my_actions
    global sample_size
    global w_me
    global w_opp
    global pwned
    
    if observation.step == 0:
        action = 1
        pwned = False
        my_actions.append(action)
        return action
    
    def increment_wins():
        global w_me
        global w_opp
        
        if my_actions[-1] == (opponent_observations[-1] + 1) % configuration.signs:
            w_me += 1
        
        elif opponent_observations[-1] == (my_actions[-1] + 1) % configuration.signs:
            w_opp += 1
        
        else:
            pass
    
    if observation.step <= 2 * sample_size:
        opponent_observations.append(observation.lastOpponentAction)
        
        increment_wins()
        
        seed = random.randint(1, 500)
        random.seed(seed)
        
        action = random.randint(0, configuration.signs-1)
        my_actions.append(action)
        return action
        
    else:
        opponent_observations.append(observation.lastOpponentAction)
        
        increment_wins()
        
        prob_pwned = was_i_pwned(observation.step, w_me=w_me, w_opp=w_opp, samples=10000, seed=42)
    
        if observation.step >= 15 and prob_pwned >= 0.75:
            pwned = True

        if pwned:
            seed = random.randint(1, 1000)
            random.seed(seed)
            action = random.randint(0, configuration.signs-1)
        
        else:
            action = action_decision_tree_dev(
                opponent_observations,
                my_actions,
                sample_size,
                configuration
            )
        
        my_actions.append(action)
        return action
