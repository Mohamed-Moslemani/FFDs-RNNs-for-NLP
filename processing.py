import re
import pickle
from collections import Counter


def extract_words_from_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
    words = re.findall(r'\b\w+\b', text)
    return words


def decapitalizer(words):
    return [word.lower() for word in words]


def popWords(words, k=10000):  # ↑ increased vocab size
    counts = Counter(words)
    return counts.most_common(k)


def replace_rare_words(lower_words, top_k_words, min_freq=3):  # ↓ lowered min_freq
    word_counts = Counter(lower_words)
    top_words_set = set(word for word, _ in top_k_words)
    rare_words = {word for word, count in word_counts.items() if count < min_freq and word not in top_words_set}
    return ["<OOV>" if word in rare_words else word for word in lower_words]


def convert_to_indices(tokenized_corpus, top_k_words):
    vocab_list = [word for word, _ in top_k_words]
    word_to_index = {word: idx + 1 for idx, word in enumerate(vocab_list)}  # reserve 0 for <UNK>
    word_to_index["<OOV>"] = 0

    index_to_word = {idx + 1: word for idx, word in enumerate(vocab_list)}
    index_to_word[0] = "<OOV>"

    indexed_corpus = [word_to_index.get(word, 0) for word in tokenized_corpus]
    return indexed_corpus, word_to_index, index_to_word


def build_context_target(corpus, context_size=4):
    X, Y = [], []
    for i in range(context_size, len(corpus)):
        X.append(corpus[i - context_size:i])
        Y.append(corpus[i])
    return X, Y


# === MAIN ===
file_path = "shakespeare.txt"
raw_words = extract_words_from_file(file_path)
lower_words = decapitalizer(raw_words)
top_k_words = popWords(lower_words, k=10000)
processed_corpus = replace_rare_words(lower_words, top_k_words, min_freq=3)
indexed_corpus, word_to_index, index_to_word = convert_to_indices(processed_corpus, top_k_words)
X, Y = build_context_target(indexed_corpus, context_size=4)
vocab_size = max(indexed_corpus) + 1

with open("dataset2.pkl", "wb") as f:
    pickle.dump({
        'X': X,
        'Y': Y,
        'vocab_size': vocab_size,
        'index_to_word': index_to_word,
    }, f)

print("✅ dataset2.pkl saved with vocab size", vocab_size)
