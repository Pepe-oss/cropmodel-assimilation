from envs import envs2
from stable_baselines3 import PPO
from stable_baselines3.common.logger import configure

test = envs2.PcseRLAssimilationEnv(
    file_config = './input2017',
    seed = 123
)

test.reset()

#%%
tmp_path = ".\\tmp\\env2"
# set up logger
new_logger = configure(tmp_path, ["stdout", "csv", "tensorboard"])
model = PPO('MlpPolicy',
            test,
            batch_size=2048,
            gae_lambda=0.90,
            gamma=0.90,
            n_epochs=100,
            ent_coef=0.01,
            verbose=1,)

# Set new logger
model.set_logger(new_logger)
model.learn(total_timesteps=1e6)  # 训练10000步

print("Training complete")

#%%
model.save("./model/env2_new.pkl")

print("Model saved")