import gymnasium as gym
from gymnasium.envs.registration import register
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import A2C, DDPG, DQN, PPO, SAC, TD3

import torch
import logging, warnings

logging.basicConfig(
    filename = 'train.log',
    filemode = 'w',
    format = '[%(asctime)s] %(message)s',
    datefmt = '%H:%M:%S',
    level = logging.INFO
)
warnings.filterwarnings("ignore")

register(
    id='rls-v0',
    entry_point='envs:RLS'
)

# Set hyper params (configurations) for training
my_config = {
    "run_id": "example",

    "algorithm": PPO,
    "policy_network": "MlpPolicy",

    "epoch_num": 10000,
    "timesteps_per_epoch": 30,
    "eval_episode_num": 40,
}

current_state_complexity = 0
best_score, best_gate_cnt = -torch.inf, torch.inf

def log(msg):
    print(msg)
    logging.info(msg)

def eval(env, model, config):
    global current_state_complexity, best_score, best_gate_cnt
    avg_score, avg_cnt = 0, 0
    for i in range(config["eval_episode_num"]):
        done = False
        score, cnt = 0, 0
            
        env.seed(i)
        obs = env.reset()

        while not done:
            action, state = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            score += reward[0]
            cnt += 1
        avg_score += score / config["eval_episode_num"]
        avg_cnt += cnt / config["eval_episode_num"]
        if i % 10 == 0:
            if current_state_complexity < info[0]["state_complexity"]:
                current_state_complexity = info[0]["state_complexity"]
                best_score, best_gate_cnt = -torch.inf, torch.inf
            # ISC = Initial State Complexity, TS = Terminal State
            log(f'Episode: {i+1:2}, ISC: {current_state_complexity:2}, Score: {score:6}, Gate: {cnt:3}, TS: {info[0]["terminal_observation"]}')
    return avg_score, avg_cnt

def train(env, model, config):
    global best_score, best_gate_cnt
    best_epoch = 0

    for epoch in range(config["epoch_num"]):

        ### Train agent using SB3
        # Uncomment to enable wandb logging
        model.learn(
            total_timesteps=config["timesteps_per_epoch"],
            reset_num_timesteps=False
        )

        ### Evaluation
        log(f'Epoch: {epoch}')
        log(f'Current best score: {best_score}, best gate count: {best_gate_cnt} in Epoch {best_epoch}')
        
        avg_score, avg_gate_cnt = eval(env, model, config)
        avg_score, avg_gate_cnt = round(avg_score, 4), round(avg_gate_cnt, 4)
        log(f'Avg_score: {avg_score}, Avg_gate_count: {avg_gate_cnt}')

        ### Save best model
        if best_score < avg_score:
            log("Saving Model")
            best_score = avg_score
            best_gate_cnt = avg_gate_cnt
            best_epoch = epoch
            model.save(f"models/best {current_state_complexity}")

        log("---------------")


if __name__ == "__main__":
    env = DummyVecEnv([lambda: gym.make('rls-v0')])

    # Create model from loaded config and train
    # Note: Set verbose to 0 if you don't want info messages
    model = my_config["algorithm"](
        my_config["policy_network"], 
        env,
        verbose = 0,
        learning_rate = 1e-3
    )
    train(env, model, my_config)