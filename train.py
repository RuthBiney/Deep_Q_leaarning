import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CheckpointCallback

# Set up the Breakout environment
env = gym.make('ALE/Breakout-v5')  # Use Breakout-v4 for compatibility

# Define the DQN agent with a convolutional neural network (CNN) policy
model = DQN(
    "CnnPolicy",  # CNN for image-based input
    env,
    learning_rate=1e-4,  # Learning rate for the optimizer
    buffer_size=50000,  # Replay buffer size
    learning_starts=1000,  # Steps before training starts
    batch_size=32,  # Size of training batch
    gamma=0.99,  # Discount factor
    train_freq=4,  # Update frequency
    target_update_interval=1000,  # Frequency of target network updates
    exploration_fraction=0.1,  # Fraction of steps to explore
    exploration_final_eps=0.01,  # Final value of epsilon for exploration
    verbose=1  # Show training details
)

# Save checkpoints during training
checkpoint_callback = CheckpointCallback(
    save_freq=10000, save_path="./checkpoints", name_prefix="breakout_dqn")

# Train the model for 50,000 steps
model.learn(total_timesteps=50000, callback=checkpoint_callback)

# Save the trained model
model.save("policy.h5")

# Close the environment
env.close()
print("Training complete. Model saved as policy.h5.")
