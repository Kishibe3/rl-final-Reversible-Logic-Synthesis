import warnings
import gymnasium as gym
from gymnasium.envs.registration import register
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import A2C, DDPG, DQN, PPO, SAC, TD3

import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

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

    "epoch_num": 1000,
    "timesteps_per_epoch": 5000,
    "eval_episode_num": 40,
}

def eval(env, model, config):
    avg_score = 0
    for i in range(config["eval_episode_num"]):
        done = False
        score, cnt = 0, 0
            
        env.seed(i)
        obs = env.reset()

        while not done:
            action, state = model.predict(obs, deterministic=True)
            obs, reward, done, _ = env.step(action)
            score += reward[0]
            cnt += 1
        avg_score += score / config["eval_episode_num"]
        if i % 10 == 0:
            print(f'Episode: {i+1}, Score: {score}, Gate: {cnt}')
    return avg_score

def train(env, model, config):
    current_best = -torch.inf
    best_epoch = 0

    for epoch in range(config["epoch_num"]):

        ### Train agent using SB3
        # Uncomment to enable wandb logging
        model.learn(
            total_timesteps=config["timesteps_per_epoch"],
            reset_num_timesteps=False,
            # callback=WandbCallback(
            #     gradient_save_freq=100,
            #     verbose=2,
            # ),
        )

        ### Evaluation
        print("Epoch: ", epoch)
        print(f'Current best score: {current_best} in Epoch {best_epoch}')
        
        avg_score = round(eval(env, model, config), 4)
        print(f'the average score is {avg_score}')

        ### Save best model
        if current_best < avg_score:
            print("Saving Model")
            current_best = avg_score
            best_epoch = epoch
            model.save(f"models/best")

        print("---------------")


if __name__ == "__main__":

    # Create wandb session (Uncomment to enable wandb logging)
    # run = wandb.init(
    #     project="assignment_3",
    #     config=my_config,
    #     sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
    #     id=my_config["run_id"]
    # )

    env = DummyVecEnv([lambda: gym.make('rls-v0')])

    # Create model from loaded config and train
    # Note: Set verbose to 0 if you don't want info messages
    model = my_config["algorithm"](
        my_config["policy_network"], 
        env,
        verbose=0,
        learning_rate=1e-4,
        #tensorboard_log=my_config["run_id"]
    )
    train(env, model, my_config)