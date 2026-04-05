#!/usr/bin/env python3
"""
QUICK START GUIDE - QuantitativeTrading-v1

Copy & paste examples to get started immediately.
"""

# ============================================================================
# EXAMPLE 1: Simplest Possible Usage (5 lines of code)
# ============================================================================

from env import TradingEnvironment, MarketDataLoader
from tasks import TaskFactory
from agents import RandomAgent

task = TaskFactory.create_easy_task()
env = TradingEnvironment(task.config, MarketDataLoader())
agent = RandomAgent(task.config.symbols)
state, _ = env.reset(seed=42)

for _ in range(50):
    action = agent.act(state, {})
    state, reward, done, info = env.step(action)
    if done:
        break

print(f"Final portfolio: ${info['portfolio_value']:.2f}")


# ============================================================================
# EXAMPLE 2: Full Training Run with Evaluation
# ============================================================================

from env import TradingEnvironment, MarketDataLoader, Config
from tasks import TaskFactory, Evaluator
from agents import RandomAgent, MomentumAgent, TechnicalAgent

def run_agent_on_task(agent, task, num_episodes=3):
    """Run agent and return metrics."""
    results = []
    
    for episode in range(num_episodes):
        loader = MarketDataLoader()
        env = TradingEnvironment(task.config, loader)
        state, _ = env.reset(seed=42 + episode)
        
        portfolio_values = [env.get_portfolio_value()]
        returns = []
        done = False
        
        while not done:
            action = agent.act(state, {})
            state, reward, done, info = env.step(action)
            portfolio_values.append(env.get_portfolio_value())
            returns.append(info['metrics']['step_return'])
        
        metrics = Evaluator.evaluate_episode(portfolio_values, returns, [])
        results.append(metrics)
    
    return results

# Test all agents on medium task
task = TaskFactory.create_medium_task()

agents = {
    'Random': RandomAgent(task.config.symbols),
    'Momentum': MomentumAgent(task.config.symbols),
    'Technical': TechnicalAgent(task.config.symbols),
}

for agent_name, agent in agents.items():
    results = run_agent_on_task(agent, task, num_episodes=3)
    avg_return = sum(r.total_return for r in results) / len(results)
    avg_sharpe = sum(r.sharpe_ratio for r in results) / len(results)
    print(f"{agent_name:10}: Return={avg_return*100:6.2f}%, Sharpe={avg_sharpe:6.2f}")


# ============================================================================
# EXAMPLE 3: Custom Agent Implementation
# ============================================================================

from agents import BaseAgent
from env import Action, OrderType

class SimpleMovingAverageAgent(BaseAgent):
    """Buy on short MA above long MA, sell on cross below."""
    
    def __init__(self, symbols):
        super().__init__(symbols, "SMAAgent")
        self.prev_signal = None
    
    def act(self, state, info):
        """Generate action based on MA crossover."""
        symbol = self.symbols[0]
        
        # In a real implementation, extract MA values from state
        # For now, use a simple heuristic
        step_return = info.get('metrics', {}).get('step_return', 0)
        
        if step_return > 0.01 and not self.prev_signal:
            action = Action(symbol=symbol, order_type=OrderType.BUY, quantity=20.0)
            self.prev_signal = True
        elif step_return < -0.01 and self.prev_signal:
            action = Action(symbol=symbol, order_type=OrderType.SELL, quantity=20.0)
            self.prev_signal = False
        else:
            action = Action(symbol=symbol, order_type=OrderType.HOLD, quantity=0.0)
        
        return action
    
    def reset(self):
        self.prev_signal = None


# ============================================================================
# EXAMPLE 4: Custom Configuration and Task
# ============================================================================

from env import Config, TradingEnvironment, MarketDataLoader

# Create custom config
custom_config = Config(
    symbols=['ASSET1', 'ASSET2'],
    initial_cash=50000.0,
    max_positions=2,
    max_position_size=30.0,
    slippage_bps=2.0,
    commission_bps=2.0,
    min_order_size=1.0,
    max_leverage=2.5,
    episode_length=150,
    lookback_window=75,
)

# Create environment with custom config
loader = MarketDataLoader()
env = TradingEnvironment(custom_config, loader)

# Use with any agent
agent = RandomAgent(custom_config.symbols)
state, _ = env.reset(seed=42)

total_reward = 0
for _ in range(100):
    action = agent.act(state, {})
    state, reward, done, info = env.step(action)
    total_reward += reward
    if done:
        break

print(f"Total Reward: {total_reward:.2f}, Portfolio: ${info['portfolio_value']:.2f}")


# ============================================================================
# EXAMPLE 5: Advanced: Loading Real Data and Analysis
# ============================================================================

from env import MarketDataLoader, TradingEnvironment, Config
from tasks import TaskFactory, Evaluator
import json

# Load custom data
loader = MarketDataLoader()

# Generate synthetic data for multiple assets
for symbol in ['ASSET1', 'ASSET2', 'ASSET3']:
    loader.generate_synthetic_data(
        symbol, 
        num_bars=2000, 
        volatility=0.02,
        trend=0.0001
    )

# Get task and environment
task = TaskFactory.create_medium_task()
env = TradingEnvironment(task.config, loader)

# Run episode
state, info = env.reset(seed=42)
agent = RandomAgent(task.config.symbols)

metrics_history = []
portfolio_values = [env.get_portfolio_value()]
returns_list = []

done = False
while not done:
    action = agent.act(state, info)
    state, reward, done, info = env.step(action)
    
    portfolio_values.append(env.get_portfolio_value())
    returns_list.append(info['metrics']['step_return'])
    metrics_history.append(info['metrics'])

# Evaluate
final_metrics = Evaluator.evaluate_episode(
    portfolio_values, 
    returns_list, 
    []
)

print(f"\nEpisode Analysis:")
print(f"  Duration: {len(portfolio_values)-1} steps")
print(f"  Return: {final_metrics.total_return*100:.2f}%")
print(f"  Sharpe: {final_metrics.sharpe_ratio:.2f}")
print(f"  Max Drawdown: {final_metrics.max_drawdown*100:.2f}%")
print(f"  Win Rate: {final_metrics.win_rate*100:.2f}%")
print(f"  Score: {final_metrics.score():.1f}/100")

# Save results
results = {
    'portfolio_values': portfolio_values,
    'returns': returns_list,
    'final_metrics': {
        'total_return': float(final_metrics.total_return),
        'sharpe_ratio': float(final_metrics.sharpe_ratio),
        'max_drawdown': float(final_metrics.max_drawdown),
        'score': float(final_metrics.score()),
    }
}

with open('episode_results.json', 'w') as f:
    json.dump(results, f, indent=2)


# ============================================================================
# EXAMPLE 6: Parallel Testing with Multiple Seeds
# ============================================================================

from env import TradingEnvironment, MarketDataLoader
from tasks import TaskFactory
from agents import MomentumAgent
import numpy as np

task = TaskFactory.create_hard_task()
seeds = [42, 123, 456, 789, 999]
results = []

for seed in seeds:
    loader = MarketDataLoader()
    env = TradingEnvironment(task.config, loader)
    agent = MomentumAgent(task.config.symbols)
    
    state, _ = env.reset(seed=seed)
    portfolio_values = [env.get_portfolio_value()]
    
    done = False
    while not done:
        action = agent.act(state, {})
        state, reward, done, info = env.step(action)
        portfolio_values.append(env.get_portfolio_value())
    
    final_return = (portfolio_values[-1] - 100000) / 100000
    results.append(final_return)

print(f"\nRobustness Analysis (5 seeds):")
print(f"  Mean Return: {np.mean(results)*100:.2f}%")
print(f"  Std Dev: {np.std(results)*100:.2f}%")
print(f"  Min: {np.min(results)*100:.2f}%")
print(f"  Max: {np.max(results)*100:.2f}%")


# ============================================================================
# EXAMPLE 7: Integration with RL Libraries (Pseudocode)
# ============================================================================

"""
# Using with Stable-Baselines3 (install: pip install stable-baselines3)

from stable_baselines3 import PPO
from env import TradingEnvironment, MarketDataLoader
from tasks import TaskFactory
import gymnasium as gym
from gymnasium import spaces
import numpy as np

class GymWrapper(gym.Env):
    '''Wrap TradingEnvironment for gym compatibility'''
    
    def __init__(self, task):
        self.env = TradingEnvironment(task.config, MarketDataLoader())
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(64,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(
            len(self.env.config.symbols) * 3  # symbol x [BUY, SELL, HOLD]
        )
    
    def reset(self, seed=None):
        state, _ = self.env.reset(seed=seed)
        return state, {}
    
    def step(self, action_idx):
        # Convert action index to Action object
        num_symbols = len(self.env.config.symbols)
        symbol_idx = action_idx // 3
        order_type_idx = action_idx % 3
        
        from env import Action, OrderType
        symbol = self.env.config.symbols[symbol_idx]
        order_types = [OrderType.BUY, OrderType.SELL, OrderType.HOLD]
        
        action = Action(
            symbol=symbol,
            order_type=order_types[order_type_idx],
            quantity=25.0
        )
        
        state, reward, done, _, info = self.env.step(action)
        return state, reward, done, False, info

# Train agent
task = TaskFactory.create_medium_task()
env = GymWrapper(task)

model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10000)

# Test trained model
obs, _ = env.reset()
for _ in range(250):
    action, _ = model.predict(obs)
    obs, reward, done, _, _ = env.step(action)
    if done:
        break
"""


# ============================================================================
# EXAMPLE 8: Debugging and Logging
# ============================================================================

import logging

# Setup detailed logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

from env import TradingEnvironment, MarketDataLoader
from tasks import TaskFactory
from agents import RandomAgent

task = TaskFactory.create_easy_task()
loader = MarketDataLoader()
env = TradingEnvironment(task.config, loader)
agent = RandomAgent(task.config.symbols)

state, info = env.reset(seed=42)

print("Starting episode debug run...")
for step in range(20):
    action = agent.act(state, info)
    state, reward, done, info = env.step(action)
    
    print(f"\nStep {step}:")
    print(f"  Action: {action.symbol} {action.order_type.name} x{action.quantity}")
    print(f"  Reward: {reward:.4f}")
    print(f"  Portfolio: ${info['portfolio_value']:.2f}")
    print(f"  Positions: {list(info['positions'].keys())}")
    
    if done:
        print(f"\nEpisode ended at step {step}")
        break


# ============================================================================
# TESTING CHECKLIST
# ============================================================================

"""
Before deploying your agent:

✓ Run quick tests:
  python test_quick.py

✓ Verify imports:
  from env import *
  from agents import *
  from tasks import *

✓ Test all difficulty levels:
  - TaskFactory.create_easy_task()
  - TaskFactory.create_medium_task()
  - TaskFactory.create_hard_task()

✓ Test all baseline agents:
  - RandomAgent
  - MomentumAgent
  - TechnicalAgent

✓ Test with multiple seeds:
  env.reset(seed=42)
  env.reset(seed=123)
  env.reset(seed=456)

✓ Run full episode:
  state, _ = env.reset()
  done = False
  while not done:
      action = agent.act(state, info)
      state, reward, done, info = env.step(action)

✓ Check metrics computation:
  metrics = Evaluator.evaluate_episode(...)
  score = metrics.score()

✓ Verify no errors:
  python -c "import env; import agents; import tasks; print('✓ OK')"
"""


if __name__ == "__main__":
    print("QuantitativeTrading-v1 - Quick Start Examples")
    print("Copy-paste any example to get started!")
    print("\nAll examples are self-contained and runnable.")
