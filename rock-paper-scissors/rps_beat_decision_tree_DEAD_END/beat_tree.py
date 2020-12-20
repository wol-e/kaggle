
"""
This agent uses a decision tree to predict it's own action, in order to guess what the opponent would predict,
and then returns the counter of the counter in order beat the oponents best action. Pretty meta...
"""

import random
import numpy as np

from sklearn.model_selection import TimeSeriesSplit
from sklearn.tree import DecisionTreeClassifier

sample_size = 5 # 5 for mirroring te opponent tree
tree_depth = 4 # 4 for mirroring the opponent tree
opponent_observations = []
my_actions = []

def make_train_set_v2(opponent_observations, my_actions, sample_size):
    observed = np.array([list(x) for x in zip(my_actions, opponent_observations)])
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

def action_beat_tree(opponent_observations, my_actions,
    sample_size, configuration=None):

    X, Y, prediction_sample = make_train_set_v2(opponent_observations, my_actions, sample_size)

    model = DecisionTreeClassifier(
        criterion="entropy",
        max_depth = tree_depth,
    )

    model.fit(X, Y)

    prediction = model.predict(prediction_sample)
    
    # action is what beats the opponents prediciton
    action = (prediction + 2) % (configuration.signs if configuration else 3)
    action = int(action[0])
        
    return action

def beat_tree_agent(observation, configuration):
    global opponent_observations
    global my_actions
    global sample_size
    
    if observation.step == 0:
        action = 1
        my_actions.append(action)
        return action
    
    elif observation.step <= 2 * sample_size:
        opponent_observations.append(observation.lastOpponentAction)
        action = random.randint(0, configuration.signs-1)
        my_actions.append(action)
        return action
        
    else:
        opponent_observations.append(observation.lastOpponentAction)
        
        action = action_beat_tree(
            opponent_observations,
            my_actions,
            sample_size,
            configuration
        )
        
        my_actions.append(action)
        return action
