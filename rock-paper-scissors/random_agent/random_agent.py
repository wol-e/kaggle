
import random

def random_agent(observation, configuration):
    seed = random.randint(1, 1000)
    random.seed(seed)
    return random.randint(0, configuration.signs-1)
