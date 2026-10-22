import numpy as np
import random
from collections import deque


class DQNAgent:
    def __init__(self, state_size=2, action_size=2, learning_rate=0.0007):
        self.state_size = state_size        # battery, queue -> 2 numbers
        self.action_size = action_size      # 0 = sleep, 1 = send
        self.lr = learning_rate

        self.gamma = 0.95              # how much future reward matters
        self.epsilon = 1.0             # exploration rate (starts high = tries random stuff)
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.995

        self.memory = deque(maxlen=5000)   # stores past experiences to learn from

        self.train_step_count = 0
        self.target_update_every = 25      # how often to refresh the target network

        # A small neural network: state -> hidden layer -> action scores
        hidden_size = 16
        self.W1 = np.random.randn(state_size, hidden_size) * 0.3
        self.b1 = np.zeros(hidden_size)
        self.W2 = np.random.randn(hidden_size, action_size) * 0.3
        self.b2 = np.zeros(action_size)

        # target network = a copy used only to compute stable training targets
        self.target_W1 = self.W1.copy()
        self.target_b1 = self.b1.copy()
        self.target_W2 = self.W2.copy()
        self.target_b2 = self.b2.copy()

    def _normalize(self, state):
        """Rescale [battery, queue] so both numbers are roughly in [0, 1] range."""
        battery, queue = state
        return np.array([battery / 100.0, queue / 10.0], dtype=np.float32)

    def _forward(self, state, use_target=False):
        """Pass the (already normalized) state through the network to get Q-values."""
        if use_target:
            W1, b1, W2, b2 = self.target_W1, self.target_b1, self.target_W2, self.target_b2
        else:
            W1, b1, W2, b2 = self.W1, self.b1, self.W2, self.b2

        z1 = state.dot(W1) + b1
        h1 = np.maximum(0, z1)              # ReLU activation
        q_values = h1.dot(W2) + b2
        return q_values, h1

    def act(self, state):
        """
        Decide an action for a given state.
        state: [battery, queue] as a list
        """
        norm_state = self._normalize(state)

        # Exploration: sometimes try a random action to discover new strategies
        if random.random() < self.epsilon:
            return random.randint(0, self.action_size - 1)

        # Exploitation: use the network's best guess
        q_values, _ = self._forward(norm_state)
        return int(np.argmax(q_values))

    def remember(self, state, action, reward, next_state, done):
        """Save this experience so the agent can learn from it later."""
        self.memory.append((state, action, reward, next_state, done))

    def _update_target_network(self):
        """Copy the main network's weights into the target network."""
        self.target_W1 = self.W1.copy()
        self.target_b1 = self.b1.copy()
        self.target_W2 = self.W2.copy()
        self.target_b2 = self.b2.copy()

    def train_step(self, batch_size=32):
        """
        Learn from a random batch of past experiences.
        This is what actually improves the agent over time.
        """
        if len(self.memory) < batch_size:
            return  # not enough experience yet

        batch = random.sample(self.memory, batch_size)

        for state, action, reward, next_state, done in batch:
            norm_state = self._normalize(state)
            norm_next_state = self._normalize(next_state)

            q_values, h1 = self._forward(norm_state)
            target = q_values.copy()

            if done:
                target[action] = reward
            else:
                # target network gives a stable estimate of future value
                next_q, _ = self._forward(norm_next_state, use_target=True)
                target[action] = reward + self.gamma * np.max(next_q)

            # --- simple gradient descent update (manual backpropagation) ---
            error = q_values - target                      # shape: (action_size,)

            grad_W2 = np.outer(h1, error)
            grad_b2 = error

            grad_h1 = error.dot(self.W2.T)
            grad_h1[h1 <= 0] = 0                            # ReLU derivative

            grad_W1 = np.outer(norm_state, grad_h1)
            grad_b1 = grad_h1

            # --- gradient clipping: stops updates from ever getting too large,
            # which is what was causing the numbers to "explode" (overflow) ---
            clip_value = 5.0
            grad_W2 = np.clip(grad_W2, -clip_value, clip_value)
            grad_b2 = np.clip(grad_b2, -clip_value, clip_value)
            grad_W1 = np.clip(grad_W1, -clip_value, clip_value)
            grad_b1 = np.clip(grad_b1, -clip_value, clip_value)

            self.W2 -= self.lr * grad_W2
            self.b2 -= self.lr * grad_b2
            self.W1 -= self.lr * grad_W1
            self.b1 -= self.lr * grad_b1

            # safety net: if numbers somehow still blow up (NaN/inf), reset them
            # to small random values instead of letting the agent stay broken
            if not np.isfinite(self.W1).all() or not np.isfinite(self.W2).all():
                self.W1 = np.random.randn(*self.W1.shape) * 0.3
                self.b1 = np.zeros_like(self.b1)
                self.W2 = np.random.randn(*self.W2.shape) * 0.3
                self.b2 = np.zeros_like(self.b2)
                self._update_target_network()

        self.train_step_count += 1
        if self.train_step_count % self.target_update_every == 0:
            self._update_target_network()

        # slowly reduce exploration as the agent learns more
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


# ---- Quick test: run this file directly with "python dqn_agent.py" ----
if __name__ == "__main__":
    agent = DQNAgent()

    # fake a few experiences and see if training runs without errors
    for _ in range(50):
        state = [random.uniform(0, 100), random.randint(0, 5)]
        action = agent.act(state)
        reward = random.uniform(-1, 1)
        next_state = [random.uniform(0, 100), random.randint(0, 5)]
        done = random.random() < 0.1
        agent.remember(state, action, reward, next_state, done)

    agent.train_step(batch_size=16)
    print("Agent created and trained one step successfully.")
    print("Current exploration rate (epsilon):", round(agent.epsilon, 3))