import torch
import numpy as np
import matplotlib.pyplot as plt
from torch import nn

dw = 0
phi = 0
om_v0 = 1
dt = 0.001

u0 = torch.tensor([[1, 0], [0, 1]]) * (1. + 0j)

def H(s_t, T):
    t = T*dt
    y = -0.5 * om_v0 * s_t * torch.tensor([[0, torch.exp(torch.tensor(+1j * (dw*t + phi)))], 
                                           [torch.exp(torch.tensor(-1j * (dw*t + phi))), 0]])
    return y


a = 1
class NeuralNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fcn = nn.Sequential(nn.Linear(a*8, a*16),
                                 nn.ReLU(),
                                 nn.Linear(a*16, a*16),
                                 nn.ReLU(),
                                 nn.Linear(a*16, a*8),
                                 nn.ReLU())  # ReLU가 무엇이지??
    
    def forward(self, x):
        y = self.fcn(x)
        return y
    

model = NeuralNet()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)  # lr = learning rate


loss_list = []
list = []
# list_str = []

epoch_range = 1000+1

for epoch in range(epoch_range):
    input_t = torch.arange(a*8)*dt
    st = model(input_t)
    model.train()

    ut = u0.detach().clone()

    for T in range(1, a*8):
        dudt = -1j*torch.matmul(H(st[T], T), ut)
        ut = ut + dudt*dt
    
    p = torch.square(torch.matmul(torch.matmul(torch.tensor([[0, 1]]) * (1. + 0j), ut), torch.tensor([[1], [0]])*(1. + 0j)).abs())[0]
    p_scalar = p.sum()
    # print("p:", p.size(), "st:", st.size())
    loss = -torch.log(p_scalar).requires_grad_(True)
    # print("log:", loss.size())
    # loss = torch.sum(st**2).requires_grad_(True)
    # print("sum:", loss.size())
    # print()
    loss.backward()
    loss_list.append(loss.item())
    optimizer.step()
    optimizer.zero_grad()

    # plt.plot(T, loss[0])
    # plt.show()
    if epoch % 10 == 1:
        print(loss)
    #     print(ut)
    #     plt.plot(st.detach().numpy(), 'b.')
    #     plt.show()
        
        list.append(loss.item())
        # list_str.append(str(loss.item()))
        

plt.plot(list, 'b.')
plt.show()
# print(list_str)