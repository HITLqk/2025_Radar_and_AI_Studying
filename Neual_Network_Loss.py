import torch
import torch.nn as nn
import torch.optim as optim

# 定义一个三层全连接网络
model = nn.Sequential(
    nn.Linear(10, 64),
    nn.ReLU(),
    nn.Linear(64, 1),
    nn.Sigmoid()
)

# 定义损失函数与优化器
criterion = nn.BCELoss()       # 二分类交叉熵
optimizer = optim.SGD(model.parameters(), lr=0.1)

# 前向传播
X = torch.randn(32, 10)        # batch_size = 32
y = torch.randint(0, 2, (32, 1)).float()
y_pred = model(X)

# 计算损失
loss = criterion(y_pred, y)

# 反向传播
optimizer.zero_grad()
loss.backward()
optimizer.step()

print("当前批次损失:", loss.item())
