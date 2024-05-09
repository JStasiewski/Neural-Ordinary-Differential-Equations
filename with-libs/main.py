import torch
import torch.nn as nn
from torchdiffeq import odeint
import matplotlib.pyplot as plt

# Define the ODE function
class ODEFunc(nn.Module):
    def __init__(self):
        super(ODEFunc, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 50),
            nn.Tanh(),
            nn.Linear(50, 2)
        )
    
    def forward(self, t, h):
        return self.net(h)

# Define the ODE Block
class ODEBlock(nn.Module):
    def __init__(self, ode_func):
        super(ODEBlock, self).__init__()
        self.ode_func = ode_func
    
    def forward(self, h0, t):
        out = odeint(self.ode_func, h0, t, method='dopri5')
        return out

# Create a complete model including the ODE block
class ODEModel(nn.Module):
    def __init__(self):
        super(ODEModel, self).__init__()
        self.ode_block = ODEBlock(ODEFunc())
    
    def forward(self, h0, t):
        return self.ode_block(h0, t)

# Instantiate the model, define loss function and optimizer
model = ODEModel()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Dummy data for demonstration
t = torch.linspace(0., 2., 100)  # Time points
h0 = torch.tensor([[0.1, 0.2]], requires_grad=True)  # Initial condition
target = torch.sin(t)  # Target for demonstration purposes

# Training loop
for epoch in range(1000):
    optimizer.zero_grad()
    h = model(h0, t)
    loss = criterion(h[:, 0, :], target.unsqueeze(1))
    loss.backward()
    optimizer.step()
    
    if epoch % 100 == 0:
        print(f'Epoch {epoch}: loss = {loss.item()}')

# Saving the entire model
torch.save(model, 'ode_model.pth')

# model = torch.load('ode_model.pth')

# Evaluate the model
with torch.no_grad():
    h_pred = model(h0, t)
    print(f'Predicted trajectory: {h_pred}')

# Extract predicted values for plotting
predicted_trajectory = h_pred[:, 0, :].detach().numpy()  # Make sure to detach and convert to numpy

# Plot the target and predicted trajectories
plt.figure(figsize=(10, 5))
plt.plot(t.numpy(), target.numpy(), label='Target sin(t)')
plt.plot(t.numpy(), predicted_trajectory[:, 0], label='Predicted Trajectory', linestyle='--')
plt.xlabel('Time')
plt.ylabel('Value')
plt.title('Comparison of Target and Predicted Trajectories')
plt.legend()
plt.show()
