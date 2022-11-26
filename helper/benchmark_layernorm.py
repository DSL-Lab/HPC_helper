import time
import torch
import torch.nn as nn
from torch.profiler import tensorboard_trace_handler


class TestNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc_in = nn.Linear(3, 128)
        self.layers = nn.ModuleList()
        for i in range(0, 10):
            self.layers.append(nn.Linear(128, 128))
            self.layers.append(nn.LayerNorm(128))
            # self.layers.append(nn.BatchNorm1d(262144))
        self.fc_out = nn.Linear(128, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc_in(x)
        for layer in self.layers:
            x = layer(x)
            # print(x.shape)
        x = self.fc_out(x)
        return x


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = TestNet().to(device).train()
in_data = torch.zeros([1, 512 * 512, 3]).to(device)
criterion = nn.MSELoss().to(device)
optimizer = torch.optim.SGD(net.parameters(), 0.01)

with torch.profiler.profile(
        schedule=torch.profiler.schedule(wait=2, warmup=2, active=6, repeat=1, skip_first=2),
        on_trace_ready=tensorboard_trace_handler("tmp/profile"),
        with_stack=True, with_flops=True, with_modules=True, profile_memory=True) as profiler:

    for i in range(0, 20):
        t0 = time.time()
        out_data = net(in_data)
        loss = criterion(out_data, in_data)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        profiler.step()
        print(f"step: {i:,d} {time.time() - t0:.3f}")
print("Done! ")