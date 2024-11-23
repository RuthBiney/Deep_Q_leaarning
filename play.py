import gym
from stable_baselines3 import DQN

# Load the trained model
model = DQN.load("policy.h5")

# Set up the Breakout environment
env = gym.make('BreakoutNoFrameskip-v4', render_mode='human')  # Use the updated environment name

# Play the game for a few episodes
for episode in range(5):
    obs, info = env.reset()  # Extract observation from the reset tuple
    done = False
    total_reward = 0

    while not done:
        # Render the environment
        env.render()

        # Get the action using the greedy policy
        action, _ = model.predict(obs, deterministic=True)

        # Take the action in the environment
        obs, reward, terminated, truncated, info = env.step(action)

        # Combine terminated and truncated flags
        done = terminated or truncated

        total_reward += reward

    print(f"Episode {episode + 1} finished with total reward: {total_reward}")

env.close()
