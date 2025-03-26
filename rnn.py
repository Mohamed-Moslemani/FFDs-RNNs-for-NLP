# rnn.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class RNNModel(nn.Module):
    def __init__(self, vocab_size, context_size, hidden_size):
        super(RNNModel, self).__init__()
        self.hidden_size = hidden_size
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.rnn = nn.RNN(hidden_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden=None):
        x = self.embed(x)
        out, hidden = self.rnn(x, hidden)
        out = self.fc(out[:, -1, :])  # take last output only
        return out, hidden

    def predict(self, x):
        self.eval()
        with torch.no_grad():
            logits, _ = self.forward(x)
            probs = F.softmax(logits, dim=-1)
            return torch.argmax(probs, dim=-1)

    def train_model(self, X, Y, iters=5000, lr=0.001):
        self.train()
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        loss_fn = nn.CrossEntropyLoss()

        for i in range(iters):
            idx = torch.randint(0, len(X), (1,)).item()
            x = torch.tensor([X[idx]], dtype=torch.long)
            y = torch.tensor([Y[idx]], dtype=torch.long)

            optimizer.zero_grad()
            logits, _ = self.forward(x)
            loss = loss_fn(logits, y)
            loss.backward()
            optimizer.step()

            if i % 500 == 0:
                print(f"Iter {i}, Loss: {loss.item():.4f}")

    def perplexity(self, X, Y):
        self.eval()
        loss_fn = nn.CrossEntropyLoss()
        total_loss = 0.0
        with torch.no_grad():
            for x, y in zip(X, Y):
                x_tensor = torch.tensor([x], dtype=torch.long)
                y_tensor = torch.tensor([y], dtype=torch.long)
                logits, _ = self.forward(x_tensor)
                loss = loss_fn(logits, y_tensor)
                total_loss += loss.item()
        return torch.exp(torch.tensor(total_loss / len(X))).item()
