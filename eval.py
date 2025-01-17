from envs.RLS_env import RLS

import gymnasium as gym
from gymnasium.envs.registration import register
from stable_baselines3 import A2C, DDPG, DQN, PPO, SAC, TD3
import matplotlib.pyplot as plt

from transformation_based import output2gates_basic, output2gates_bidirectional, gates2output

register(
    id='rls-eval',
    entry_point='envs:RLS_eval'
)

def eval(env, model, eval_num = 100):
    rl_gate_cnts, scores = [], []
    basic_gate_cnts, bidirectional_gate_cnts = [], []

    for seed in range(eval_num):
        done = False
        score = 0
        actions = []

        obs, _ = env.reset(seed = seed)
        init_state = obs

        while not done:
            action, _ = model.predict(obs, deterministic = True)
            actions.append(env.unwrapped.actiondict[action])
            obs, reward, done, _, _ = env.step(action)
            score += reward
        if gates2output(env.unwrapped.n, actions[::-1]) == init_state.tolist():  # give correct answer
            scores.append(score)
            rl_gate_cnts.append(len(actions))
            basic_gate_cnts.append(len(output2gates_basic(env.unwrapped.n, init_state.tolist())))
            bidirectional_gate_cnts.append(len(output2gates_bidirectional(env.unwrapped.n, init_state.tolist())))
    
    print(f'Total Trials: {eval_num}')
    print(f'Correctness: {100 * len(scores) / eval_num}%')
    print(f'Avg score: {sum(scores) / len(scores)}')
    print(f'RL avg gate count: {sum(rl_gate_cnts) / len(scores)}')
    print(f'Basic avg gate count: {sum(basic_gate_cnts) / len(scores)}')
    print(f'Bidirectional avg gate count: {sum(bidirectional_gate_cnts) / len(scores)}')

    plt.plot(list(range(1, len(scores) + 1)), rl_gate_cnts, color = 'red', label = 'RL', linestyle = ':', marker = '.')
    plt.plot(list(range(1, len(scores) + 1)), basic_gate_cnts, color = 'green', label = 'Basic', linestyle = ':', marker = '.')
    plt.plot(list(range(1, len(scores) + 1)), bidirectional_gate_cnts, color = 'blue', label = 'Bidirectional', linestyle = ':', marker = '.')
    plt.xlabel('Trials')
    plt.ylabel('Gate Counts')
    plt.title(f'Used Gate Counts, RL Correctness = {100 * len(scores) / eval_num}%')
    plt.legend()
    plt.savefig('Result.png')
    #plt.show()

if __name__ == '__main__':
    models = [
        'models/2/best 7, score=16.84, gate=9.12, match=200',  # 99.2% 打敗basic
        'models/4/best 8, score=18.5, gate=8.7, match=160'  # 97.3% 打敗bidirectional
    ]
    env = gym.make('rls-eval')

    model = PPO.load(models[0])
    eval(env, model, 10000)