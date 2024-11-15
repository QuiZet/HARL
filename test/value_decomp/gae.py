# gae.py

import numpy as np
from config import GAMMA

def compute_gae(next_value, rewards, masks, values):
    values = np.append(values, next_value)
    gae = 0
    returns = []
    for step in reversed(range(len(rewards))):
        delta = rewards[step] + GAMMA * values[step + 1] * masks[step] - values[step]
        gae = delta + GAMMA * gae * masks[step]
        returns.insert(0, gae + values[step])
    adv = np.array(returns) - values[:-1]
    return returns, adv
