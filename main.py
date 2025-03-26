import pickle
import random
import torch
from ffd import ffd
from rnn import RNNModel

# Load dataset
with open('dataset2.pkl', 'rb') as f:
    data = pickle.load(f)
    X = data['X']
    Y = data['Y']
    vocab_size = data['vocab_size']
    index_to_word = data['index_to_word']

# Context length
ctx = len(X[0])
hidden = 256

# ----------- Feedforward baseline -----------
print("⚙️ Training ffd model...")
m_ffd = ffd(vocab_size, ctx, hidden)
m_ffd.train(X, Y, iters=5000, lr=0.1)

pp_ffd = m_ffd.perplexity(X[:1000], Y[:1000])
print("📉 ffd perplexity:", pp_ffd)

sample = random.choice(X)
pred_ffd = m_ffd.predict([sample])[0]
decoded_pred_ffd = index_to_word.get(pred_ffd, "<OOV>")

# ----------- RNN model -----------
print("\n⚙️ Training RNN model...")
rnn = RNNModel(vocab_size, ctx, hidden)
rnn.train_model(X, Y, iters=5000, lr=0.001)

pp_rnn = rnn.perplexity(X[:1000], Y[:1000])
print("📉 RNN perplexity:", pp_rnn)

# Predict
x_tensor = torch.tensor([sample], dtype=torch.long)
pred_rnn = rnn.predict(x_tensor)[0].item()
decoded_pred_rnn = index_to_word.get(pred_rnn, "<OOV>")

# ----------- Show input & predictions -----------
decoded_input = [index_to_word.get(i, "<OOV>") for i in sample]

print("\n📝 input:", decoded_input)
print("🤖 ffd predicted:", decoded_pred_ffd)
print("🤖 RNN predicted:", decoded_pred_rnn)
print("📊 raw indices:", sample)
