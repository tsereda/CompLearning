import torch
import torch.nn as nn
import torch.optim as optim
import asyncio
import random
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
from collections import defaultdict

# Constants for the compositional task
COLORS = ['red', 'blue', 'green', 'yellow']
SHAPES = ['circle', 'square', 'triangle', 'star']

class SharedState:
    def __init__(self):
        self.data: Dict[str, Tuple[int, Dict[str, torch.Tensor]]] = {}
    
    def update(self, key: str, value: Dict[str, torch.Tensor], version: int) -> bool:
        if key not in self.data or self.data[key][0] < version:
            self.data[key] = (version, value)
            return True
        return False
    
    def get(self, key: str) -> Tuple[int, Dict[str, torch.Tensor]]:
        return self.data.get(key, (0, None))

class CompositionaLearningAgent(nn.Module):
    def __init__(self, id: int, input_size: int, hidden_size: int, vocab_size: int):
        super(CompositionaLearningAgent, self).__init__()
        self.id = id
        self.peers: List[int] = []
        self.state = SharedState()
        
        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size)
        )
        self.decoder = nn.Sequential(
            nn.Linear(vocab_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, input_size)
        )
        self.message_generator = nn.Linear(hidden_size, vocab_size)
        
        self.optimizer = optim.Adam(self.parameters(), lr=0.001)
        self.loss_history = []
    
    def forward(self, x, tau=1.0):
        encoded = self.encoder(x)
        message = torch.nn.functional.gumbel_softmax(self.message_generator(encoded), tau=tau, hard=False)
        decoded = self.decoder(message)
        return message, decoded
    
    async def update_local_state(self):
        version, _ = self.state.get("model_params")
        new_version = version + 1
        new_state_dict = self.state_dict()
        if self.state.update("model_params", new_state_dict, new_version):
            print(f"Agent {self.id} updated local state: model_params (v{new_version})")
            await self.propagate_update("model_params", new_state_dict, new_version)
    
    async def propagate_update(self, key: str, value: Dict[str, torch.Tensor], version: int):
        for peer_id in self.peers:
            await self.send_update(peer_id, key, value, version)
    
    async def send_update(self, to_id: int, key: str, value: Dict[str, torch.Tensor], version: int):
        print(f"Agent {self.id} sending update to Agent {to_id}: {key} (v{version})")
        await asyncio.sleep(0.1)
    
    async def receive_update(self, from_id: int, key: str, value: Dict[str, torch.Tensor], version: int):
        if self.state.update(key, value, version):
            print(f"Agent {self.id} received update from Agent {from_id}: {key} (v{version})")
            self.load_state_dict(value)
            await self.propagate_update(key, value, version)
    
    async def train_step(self, inputs: torch.Tensor, targets: torch.Tensor):
        self.optimizer.zero_grad()
        messages, outputs = self(inputs)
        loss = nn.functional.mse_loss(outputs, targets)
        loss.backward()
        self.optimizer.step()
        
        self.loss_history.append(loss.item())
        await self.update_local_state()
        
        return loss.item()

def generate_compositional_data(batch_size: int):
    inputs = torch.zeros(batch_size, len(COLORS) + len(SHAPES))
    targets = torch.zeros(batch_size, len(COLORS) + len(SHAPES))
    
    for i in range(batch_size):
        color_idx = random.randint(0, len(COLORS) - 1)
        shape_idx = random.randint(0, len(SHAPES) - 1)
        inputs[i, color_idx] = 1
        inputs[i, len(COLORS) + shape_idx] = 1
        
        # For targets, we'll randomly change either color or shape
        if random.choice([True, False]):
            new_color_idx = (color_idx + 1) % len(COLORS)
            targets[i, new_color_idx] = 1
            targets[i, len(COLORS) + shape_idx] = 1
        else:
            new_shape_idx = (shape_idx + 1) % len(SHAPES)
            targets[i, color_idx] = 1
            targets[i, len(COLORS) + new_shape_idx] = 1
    
    return inputs, targets

class DistributedLearningSimulator:
    def __init__(self):
        self.agents: Dict[int, CompositionaLearningAgent] = {}
    
    def add_agent(self, agent: CompositionaLearningAgent):
        self.agents[agent.id] = agent
    
    def setup_peers(self):
        for agent in self.agents.values():
            agent.peers = [id for id in self.agents.keys() if id != agent.id]
    
    async def simulate_network(self):
        while True:
            sender = random.choice(list(self.agents.values()))
            receiver = self.agents[random.choice(sender.peers)]
            key = "model_params"
            version, value = sender.state.get(key)
            if value is not None:
                await receiver.receive_update(sender.id, key, value, version)
            await asyncio.sleep(0.5)
    
    async def run_simulation(self, num_epochs: int, batch_size: int):
        network_task = asyncio.create_task(self.simulate_network())
        
        for epoch in range(num_epochs):
            print(f"Epoch {epoch + 1}/{num_epochs}")
            inputs, targets = generate_compositional_data(batch_size)
            
            tasks = [agent.train_step(inputs, targets) for agent in self.agents.values()]
            losses = await asyncio.gather(*tasks)
            
            avg_loss = sum(losses) / len(losses)
            print(f"Average Loss: {avg_loss:.4f}")
            
            await asyncio.sleep(1)
        
        network_task.cancel()
        await asyncio.gather(network_task, return_exceptions=True)
        print("Simulation completed.")
    
    def plot_loss_history(self):
        plt.figure(figsize=(10, 6))
        for agent_id, agent in self.agents.items():
            plt.plot(agent.loss_history, label=f"Agent {agent_id}")
        plt.title("Loss History")
        plt.xlabel("Training Step")
        plt.ylabel("Loss")
        plt.legend()
        plt.show()
    
    def analyze_messages(self):
        test_inputs, _ = generate_compositional_data(100)
        message_stats = defaultdict(lambda: defaultdict(int))
        
        for agent in self.agents.values():
            messages, _ = agent(test_inputs)
            max_indices = messages.argmax(dim=1)
            
            for i, idx in enumerate(max_indices):
                color = COLORS[test_inputs[i, :len(COLORS)].argmax().item()]
                shape = SHAPES[test_inputs[i, len(COLORS):].argmax().item()]
                message_stats[f"{color}_{shape}"][idx.item()] += 1
        
        print("\nMessage Analysis:")
        for concept, messages in message_stats.items():
            print(f"{concept}:")
            for message, count in messages.items():
                print(f"  Message {message}: {count} times")

async def main():
    simulator = DistributedLearningSimulator()
    
    for i in range(3):
        simulator.add_agent(CompositionaLearningAgent(i, input_size=len(COLORS)+len(SHAPES), hidden_size=64, vocab_size=10))
    
    simulator.setup_peers()
    
    await simulator.run_simulation(num_epochs=50, batch_size=32)
    
    for agent in simulator.agents.values():
        print(f"Agent {agent.id} final state version: {agent.state.get('model_params')[0]}")
    
    simulator.plot_loss_history()
    simulator.analyze_messages()

asyncio.run(main())