import gym
from stable_baselines3 import DQN

# Load the trained model
model = DQN.load("policy.h5")

# Set up the Breakout environment
env = gym.make('Breakout-v0')

# Reset the environment
obs = env.reset()

# Play the game for a few episodes
for episode in range(5):
    obs = env.reset()
    done = False
    total_reward = 0

    while not done:
        # Render the environment
        env.render()

        # Get the action using the greedy policy
        action, _ = model.predict(obs, deterministic=True)

        # Take the action in the environment
        obs, reward, done, info = env.step(action)
        total_reward += reward

    print(f"Episode {episode + 1} finished with total reward: {total_reward}")

env.close()
