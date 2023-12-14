from envs.RLS_env import RLS

import gymnasium as gym
from gymnasium.envs.registration import register
from stable_baselines3 import A2C, DDPG, DQN, PPO, SAC, TD3

register(
    id='rls-eval',
    entry_point='envs:RLS'
)

def evaluation(env, model, eval_num = 100):
    gate_counts, returns = [], []

    for seed in range(eval_num):
        done = False
        gate_cnt, sum_reward = 0, 0
        obs, _ = env.reset(seed = seed)

        while not done:
            action, _ = model.predict(obs, deterministic = True)
            obs, reward, done, _, _ = env.step(action)
            gate_cnt += 1
            sum_reward += reward

        gate_counts.append(gate_cnt)
        returns.append(sum_reward)
    
    print(f'Avg gate counts: {sum(gate_counts) / eval_num}')
    print(f'Avg returns: {sum(returns) / eval_num}')

if __name__ == "__main__":
    model_path = "models/best"
    env = gym.make('rls-eval')

    model = PPO.load(model_path)
    evaluation(env, model, 100)