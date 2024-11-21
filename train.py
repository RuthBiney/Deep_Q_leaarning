import gym
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CheckpointCallback

# Set up the Breakout environment
env = gym.make('Breakout-v4')

# Create a DQN model with CnnPolicy
model = DQN(
    "CnnPolicy",
    env,
    learning_rate=1e-4,
    buffer_size=50000,
    learning_starts=1000,
    batch_size=32,
    tau=1.0,
    gamma=0.99,
    target_update_interval=1000,
    train_freq=4,
    exploration_fraction=0.1,
    exploration_final_eps=0.02,
    verbose=1,
)

# Set up a callback to save the model during training
checkpoint_callback = CheckpointCallback(
    save_freq=50000, save_path='./models/', name_prefix='dqn_breakout')

# Train the model
model.learn(total_timesteps=10000, callback=checkpoint_callback)

# Save the trained model
model.save("policy.h5")

print("Training completed and model saved!")
