import gym
from stable_baselines3 import DQN

# Load the trained model
model = DQN.load("policy.h5")

# Set up the Breakout environment
env = gym.make('Breakout-v0')

# Play a few episodes with the trained agent
print("Playing with the trained agent...")
for episode in range(5):  # Play 5 episodes
    obs = env.reset()
    done = False
    total_reward = 0

    while not done:
        env.render()  # Render the game in real-time
        # Greedy action selection
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        total_reward += reward

    print(f"Episode {episode + 1}: Total Reward = {total_reward}")

env.close()
