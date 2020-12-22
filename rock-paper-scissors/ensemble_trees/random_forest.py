
import random
import numpy as np

from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier

sample_size = 5
opponent_observations = []
my_actions = []

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

    model = RandomForestClassifier(
        n_estimators=100,
        criterion="entropy",
        max_depth = 4,
    )

    model.fit(X, Y)

    prediction = model.predict(prediction_sample)
    
    # action is what beats the opponents prediciton
    action = (prediction + 1) % (configuration.signs if configuration else 3)
    action = int(action[0])
    
    # check if fails on long wait
    import time
    
    return action

def random_forest_agent(observation, configuration):
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
        
        action = action_decision_tree_dev(
            opponent_observations,
            my_actions,
            sample_size,
            configuration
        )
        
        my_actions.append(action)
        return action
