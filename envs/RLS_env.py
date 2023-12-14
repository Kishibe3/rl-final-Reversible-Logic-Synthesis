import gymnasium as gym
from gymnasium.utils import seeding
import numpy as np
from stable_baselines3.common.env_checker import check_env

from collections import defaultdict
from queue import Queue

from transformation_based import output2gates_bidirectional

class RLS(gym.Env):
    def __init__(self):
        super().__init__()
        def dfs(i, action):
            if i == self.n:
                if action.count('2') == 1:
                    self.actiondict.append(action)
                return
            # 0: no, 1: control, 2: not
            for a in range(3):
                dfs(i + 1, str(a) + action)

        self.n = 3
        self.actiondict = []
        dfs(0, '')
        self.action_space = gym.spaces.Discrete(len(self.actiondict))
        self.observation_space = gym.spaces.Box(low = 0, high = (1 << self.n) - 1, shape = (1 << self.n,), dtype = int)
        self.step_penalty = -1
        self.seed()
        self.reset()

    def step(self, action):
        self.step_cnt += 1
        
        if self.step_cnt == self.max_step:
            return self.state, -len(output2gates_bidirectional(self.state)), True, False, {}
        if (self.state == np.arange(1 << self.n)).all():
            return self.state, self.n * (1 << self.n), True, False, {}
        
        action = self.actiondict[action]
        ctrl = int(action.replace('2', '0'), 2)
        self.state = np.array([x ^ int(action.replace('1', '0').replace('2', '1'), 2) if ctrl & x == ctrl else x for x in self.state])

        # Return state, reward, done, truncate and info
        return self.state, self.step_penalty, False, False, {}
    
    def reset(self, seed=None):
        self.seed(seed=seed)
        self.step_cnt, self.max_step = 0, self.n * (1 << self.n)

        """
        也許可先從最簡單的state(即僅需少量gate就能夠造的state)開始學起，之後再學複雜的state
        """
        self.state = np.arange(1 << self.n)
        np.random.shuffle(self.state)
        return self.state, {}
    
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def synthetic_demonstrations(self):
        """
        棄用
        
        事先製造所有可能的state與所需要的gate數量，作為衡量reward的參考
        缺點: 只能用在self.n = 3以下，self.n = 4就太大了
        """
        self.sd = defaultdict(lambda: self.n * (1 << self.n) + 10)
        q = Queue()
        self.sd[*range(1 << self.n)] = 0
        q.put(tuple(range(1 << self.n)))
        while not q.empty():
            state = q.get()
            for action in self.actiondict:
                ctrl = int(action.replace('2', '0'), 2)
                new_state = tuple(x ^ int(action.replace('1', '0').replace('2', '1'), 2) if ctrl & x == ctrl else x for x in state)
                if self.sd.get(new_state) == None:
                    self.sd[new_state] = self.sd[state] + 1
                    q.put(new_state)
        