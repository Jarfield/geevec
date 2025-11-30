import numpy as np

from models import MLPClassifier
from loss import SoftmaxCrossEntropyLoss
from optim import SGD

# 1. 构造一个小 MLP
model = MLPClassifier(input_dim=4, hidden_dims=[5], num_classes=3)

# 2. 随机数据
np.random.seed(0)
X = np.random.randn(10, 4).astype(np.float32)
y = np.random.randint(0, 3, size=(10,))

loss_fn = SoftmaxCrossEntropyLoss()
optimizer = SGD(model, lr=0.1)

# 3. forward + backward + step
logits = model.forward(X)
loss = loss_fn.forward(logits, y)
grad_logits = loss_fn.backward()
model.backward(grad_logits)
optimizer.step()
optimizer.zero_grad()

print("Loss:", loss)
