import torch.nn as nn
import torch.nn.functional as F

# Encoder
class Q_net(nn.Module):
    def __init__(self, input_size=784, hidden_size=1000, z_size=2, n_classes=10):
        super(Q_net, self).__init__()
        self.lin1 = nn.Linear(input_size, hidden_size)
        self.lin2 = nn.Linear(hidden_size, hidden_size)
        # gaussian encoding (z)
        self.lin3_gauss = nn.Linear(hidden_size, z_size)
        # categorical label (y)
        self.lin3_cat = nn.Linear(hidden_size, n_classes)

    def forward(self, x):
        x = F.dropout(self.lin1(x), p=0.25, training=self.training)
        x = F.relu(x)
        x = F.dropout(self.lin2(x), p=0.25, training=self.training)
        x = F.relu(x)
        z_gauss = self.lin3_gauss(x)
        y_cat = F.softmax(self.lin3_cat(x))

        return y_cat, z_gauss


# Decoder
class P_net(nn.Module):
    def __init__(self, input_size=784, hidden_size=1000, z_size=2, n_classes=10):
        super(P_net, self).__init__()
        self.lin1 = nn.Linear(z_size + n_classes, hidden_size)
        self.lin2 = nn.Linear(hidden_size, hidden_size)
        self.lin3 = nn.Linear(hidden_size, input_size)

    def forward(self, x):
        x = self.lin1(x)
        x = F.dropout(x, p=0.25, training=self.training)
        x = F.relu(x)
        x = self.lin2(x)
        x = F.dropout(x, p=0.25, training=self.training)
        x = self.lin3(x)
        return F.sigmoid(x)


# Discriminator categorial
class D_net_cat(nn.Module):
    def __init__(self, n_classes=10, hidden_size=1000):
        super(D_net_cat, self).__init__()
        self.lin1 = nn.Linear(n_classes, hidden_size)
        self.lin2 = nn.Linear(hidden_size, hidden_size)
        self.lin3 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = self.lin1(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.lin2(x)
        x = F.relu(x)
        x = self.lin3(x)
        return F.sigmoid(x)


# Discriminator gaussian
class D_net_gauss(nn.Module):
    def __init__(self, z_size=2, hidden_size=1000):
        super(D_net_gauss, self).__init__()
        self.lin1 = nn.Linear(z_size, hidden_size)
        self.lin2 = nn.Linear(hidden_size, hidden_size)
        self.lin3 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = F.dropout(self.lin1(x), p=0.2, training=self.training)
        x = F.relu(x)
        x = F.dropout(self.lin2(x), p=0.2, training=self.training)
        x = F.relu(x)
        return F.sigmoid(self.lin3(x))
