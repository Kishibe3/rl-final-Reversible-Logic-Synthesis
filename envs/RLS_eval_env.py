import gymnasium as gym
import numpy as np

from transformation_based import output2gates_bidirectional

class RLS_eval(gym.Env):
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
        self.reset()

    def step(self, action):
        self.step_cnt += 1

        action = self.actiondict[action]
        ctrl = int(action.replace('2', '0'), 2)
        self.state = np.array([x ^ int(action.replace('1', '0').replace('2', '1'), 2) if ctrl & x == ctrl else x for x in self.state])

        if self.step_cnt == self.max_step:
            return self.state, -len(output2gates_bidirectional(self.n, self.state.tolist())), True, False, {}
        if (self.state == np.arange(1 << self.n)).all():
            return self.state, self.max_step + self.expected_gate_cnt - self.step_cnt, True, False, {}

        # Return state, reward, done, truncate and info
        return self.state, self.step_penalty, False, False, {}
    
    def reset(self, **kwargs):
        kwargs.pop('options', None)
        self.step_cnt, self.max_step = 0, self.n * (1 << self.n)

        self.state = np.arange(1 << self.n)
        np.random.shuffle(self.state)
        
        # 用於計算正確合成時的reward
        self.expected_gate_cnt = len(output2gates_bidirectional(self.n, self.state.tolist()))
        return self.state, {}
