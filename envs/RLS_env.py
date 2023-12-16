import gymnasium as gym
from gymnasium.utils import seeding
import numpy as np
from stable_baselines3.common.env_checker import check_env

from collections import defaultdict
from queue import Queue
import random

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

        self.generator = self.init_state_generator()
        self.reset()

    def step(self, action):
        self.step_cnt += 1
        
        if self.step_cnt == self.max_step:
            return self.state, -len(output2gates_bidirectional(self.n, self.state.tolist())), True, False, {}
        if (self.state == np.arange(1 << self.n)).all():
            return self.state, self.max_step + self.expected_gate_cnt - self.step_cnt, True, False, {}
        
        action = self.actiondict[action]
        ctrl = int(action.replace('2', '0'), 2)
        self.state = np.array([x ^ int(action.replace('1', '0').replace('2', '1'), 2) if ctrl & x == ctrl else x for x in self.state])

        # Return state, reward, done, truncate and info
        return self.state, self.step_penalty, False, False, {}
    
    def reset(self, seed=None):
        self.seed(seed=seed)
        self.step_cnt, self.max_step = 0, self.n * (1 << self.n)

        self.state = next(self.generator)
        
        # 用於計算正確合成時的reward
        self.expected_gate_cnt = len(output2gates_bidirectional(self.n, self.state.tolist()))
        return self.state, {}
    
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def init_state_generator(self):
        """
        從最簡單的state(即僅需少量gate就能夠造的state)開始學起，之後再學複雜的state
        """
        d = {}
        q = Queue()
        d[*range(1 << self.n)] = 0
        q.put(tuple(range(1 << self.n)))
        update = 0  # 使用update數量以下的gate製造出來的state都會被從d中刪掉
        max_gate_usage = 1
        sample_states = []

        while not q.empty():
            state = q.get()
            if d[state] - update > 1:
                d = {k: v for k, v in d.items() if v > update}  # 刪除之後永遠不會再用到的state，減少空間消耗
                update += 1
            for action in self.actiondict:
                ctrl = int(action.replace('2', '0'), 2)
                new_state = tuple(x ^ int(action.replace('1', '0').replace('2', '1'), 2) if ctrl & x == ctrl else x for x in state)
                if d.get(new_state) == None:
                    d[new_state] = d[state] + 1
                    q.put(new_state)
                    sample_states.append(np.array(new_state))
                # new_state不可能等於用少於(d[state] - 1)個gate製造出來的任何state
                # 因此前面才能安心地把部分d的內容刪掉
            if q.empty() or d[q.queue[0]] == max_gate_usage:
                max_gate_usage += 1
                for _ in range(100 * len(sample_states)):  # 在此調整要多快的速度學習更複雜的state
                    yield random.choice(sample_states)
                sample_states = []
