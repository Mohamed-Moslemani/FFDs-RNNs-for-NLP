import numpy as np
import pickle

class ffd:

    def __init__(self, vocab_size, ctx_size, hidden_sz):
        self.vocab = vocab_size
        self.ctx = ctx_size
        self.hidden = hidden_sz

        lim = np.sqrt(1. / (self.vocab * self.ctx))
        self.W1 = np.random.uniform(-lim, lim, (self.vocab * self.ctx, self.hidden))
        self.b1 = np.zeros((1, self.hidden))

        lim2 = np.sqrt(1. / self.hidden)
        self.W2 = np.random.uniform(-lim2, lim2, (self.hidden, self.vocab))
        self.b2 = np.zeros((1, self.vocab))

    def one_hot(self, idx, V):
        one = np.zeros((len(idx), V))
        for i, v in enumerate(idx):
            one[i][v] = 1
        return one

    def forward(self, x):
        self.h = np.dot(x, self.W1) + self.b1
        self.h_relu = np.maximum(0, self.h)
        self.logits = np.dot(self.h_relu, self.W2) + self.b2
        exp = np.exp(self.logits - np.max(self.logits, axis=1, keepdims=True))
        self.probs = exp / np.sum(exp, axis=1, keepdims=True)
        return self.probs

    def loss(self, y_pred, y_true):
        n = y_pred.shape[0]
        log_likelihood = -np.log(y_pred[range(n), y_true])
        loss = np.sum(log_likelihood) / n
        return loss

    def backward(self, x, y):
        n = x.shape[0]
        dlogits = self.probs
        dlogits[range(n), y] -= 1
        dlogits /= n

        dW2 = np.dot(self.h_relu.T, dlogits)
        db2 = np.sum(dlogits, axis=0, keepdims=True)

        dh_relu = np.dot(dlogits, self.W2.T)
        dh = dh_relu * (self.h > 0)

        dW1 = np.dot(x.T, dh)
        db1 = np.sum(dh, axis=0, keepdims=True)

        return dW1, db1, dW2, db2

    def update(self, dW1, db1, dW2, db2, lr):
        self.W1 -= lr * dW1
        self.b1 -= lr * db1
        self.W2 -= lr * dW2
        self.b2 -= lr * db2

    def train(self, X, Y, iters, lr):
        for i in range(iters):
            ix = np.random.randint(0, len(X), 32)
            batch_x = [X[j] for j in ix]
            batch_y = [Y[j] for j in ix]
            xhot = np.array([
            np.concatenate(self.one_hot(seq, self.vocab)) for seq in batch_x
            ])

            y_pred = self.forward(xhot)
            L = self.loss(y_pred, batch_y)
            dW1, db1, dW2, db2 = self.backward(xhot, batch_y)
            self.update(dW1, db1, dW2, db2, lr)
            if i % 100 == 0:
                print("loss:", L)

    def predict(self, x):
        xhot = np.array([
            np.concatenate(self.one_hot(seq, self.vocab)) for seq in x
        ])
        probs = self.forward(xhot)
        return np.argmax(probs, axis=1)

    def perplexity(self, X, Y):
        ppl = 0
        total = 0
        for i in range(0, len(X), 32):
            bx = X[i:i+32]
            by = Y[i:i+32]
            xhot = np.array([
                np.concatenate(self.one_hot(seq, self.vocab)) for seq in bx
            ])
            probs = self.forward(xhot)
            l = self.loss(probs, by)
            ppl += np.exp(l) * len(bx)
            total += len(bx)
        return ppl / total
