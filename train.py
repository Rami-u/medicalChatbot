import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from nltk_utils import tokenize, stem, bag_of_words
from model import NeuralNet


# ─── Step 1: Load intents.json ────────────────────────────────────────────────

with open("intents.json", "r") as f:
    intents = json.load(f)


# ─── Step 2: Extract words and tags from patterns ────────────────────────────
#
# IMPORTANT — Context system note:
# We train on ALL intents including child intents (context_filter ones).
# The model must learn to recognize words like "severe" or "one side" as
# belonging to headache_migraine, "room spinning" as dizziness_positional, etc.
# Context filtering happens at RUNTIME in chat.py — not during training.
#
# The model learns: "what tag does this input most likely belong to?"
# The chat.py context logic decides: "is this tag allowed right now?"

all_words = []   # every unique stemmed word from all patterns
tags      = []   # every unique intent tag
xy        = []   # list of (tokenized_pattern, tag) pairs

for intent in intents["intents"]:
    tag = intent["tag"]
    tags.append(tag)

    for pattern in intent["patterns"]:
        # tokenize each pattern into a list of words
        word_list = tokenize(pattern)
        all_words.extend(word_list)
        # store the pair for later training data creation
        xy.append((word_list, tag))


# ─── Step 3: Clean and deduplicate the vocabulary ────────────────────────────

# characters to ignore — not useful as features
ignore_chars = ["?", "!", ".", ",", "'", "-", "/"]

# stem every word and remove ignored characters
all_words = [stem(w) for w in all_words if w not in ignore_chars]

# remove duplicates and sort alphabetically
all_words = sorted(set(all_words))
tags      = sorted(set(tags))

print(f"Vocabulary size  : {len(all_words)} unique stemmed words")
print(f"Number of tags   : {len(tags)} intent tags")
print(f"Training samples : {len(xy)} pattern-tag pairs")
print()


# ─── Step 4: Build training data ─────────────────────────────────────────────
#
# X_train : bag-of-words vector for each pattern  shape: (N, vocab_size)
# y_train : integer tag index for each pattern    shape: (N,)

X_train = []
y_train = []

for (pattern_words, tag) in xy:
    # convert the pattern to a bag-of-words vector
    bag = bag_of_words(pattern_words, all_words)
    X_train.append(bag)

    # y is the index of the tag in the sorted tags list
    label = tags.index(tag)
    y_train.append(label)

X_train = np.array(X_train, dtype=np.float32)
y_train = np.array(y_train, dtype=np.int64)  # CrossEntropyLoss needs int64


# ─── Step 5: Create PyTorch Dataset and DataLoader ───────────────────────────

class ChatDataset(Dataset):
    """Simple dataset wrapper for our training pairs."""

    def __init__(self):
        self.n_samples = len(X_train)
        self.x_data    = X_train
        self.y_data    = y_train

    def __len__(self):
        return self.n_samples

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]


# ─── Step 6: Hyperparameters ─────────────────────────────────────────────────
#
# These values are tunable — change and retrain to compare results.
# hidden_size: try 8, 16, 32 — larger = more capacity but slower
# learning_rate: 0.001 is a reliable default for Adam
# num_epochs: 1000 is enough for this dataset size

BATCH_SIZE    = 8
HIDDEN_SIZE   = 16     # neurons per hidden layer — increased from 8 for better accuracy
LEARNING_RATE = 0.001
NUM_EPOCHS    = 1500   # more epochs for the larger intents file

INPUT_SIZE  = len(all_words)   # vocab size
OUTPUT_SIZE = len(tags)        # number of intent tags

print(f"Input size    : {INPUT_SIZE}")
print(f"Hidden size   : {HIDDEN_SIZE}")
print(f"Output size   : {OUTPUT_SIZE}")
print(f"Epochs        : {NUM_EPOCHS}")
print(f"Learning rate : {LEARNING_RATE}")
print()

dataset    = ChatDataset()
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)


# ─── Step 7: Initialize model, loss function, optimizer ──────────────────────

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training on: {device}")
print()

model = NeuralNet(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE).to(device)

# CrossEntropyLoss is best for multi-class classification
# It combines softmax + negative log likelihood internally
criterion = nn.CrossEntropyLoss()

# Adam optimizer — adaptive learning rate, works well out of the box
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)


# ─── Step 8: Training loop ───────────────────────────────────────────────────

print("Starting training...")
print("-" * 45)

for epoch in range(NUM_EPOCHS):

    total_loss = 0

    for (words, labels) in dataloader:
        words  = words.to(device)
        labels = labels.to(device)

        # forward pass — get predictions
        outputs = model(words)

        # calculate loss — how wrong is the model?
        loss = criterion(outputs, labels)

        # backward pass — calculate gradients
        optimizer.zero_grad()   # clear old gradients first
        loss.backward()         # compute new gradients

        # update weights — improve the model
        optimizer.step()

        total_loss += loss.item()

    # print progress every 100 epochs
    if (epoch + 1) % 100 == 0:
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch [{epoch+1:4d}/{NUM_EPOCHS}]  |  Loss: {avg_loss:.6f}")

print("-" * 45)
print(f"Training complete. Final loss: {avg_loss:.6f}")
print()


# ─── Step 9: Save the trained model ──────────────────────────────────────────
#
# We save everything needed to reload the model in chat.py:
#   - model_state : the trained weights
#   - input_size  : to rebuild the NeuralNet
#   - hidden_size : to rebuild the NeuralNet
#   - output_size : to rebuild the NeuralNet
#   - all_words   : the vocabulary (needed for bag_of_words at runtime)
#   - tags        : the tag list (needed to convert prediction index to tag name)

save_data = {
    "model_state": model.state_dict(),
    "input_size" : INPUT_SIZE,
    "hidden_size": HIDDEN_SIZE,
    "output_size": OUTPUT_SIZE,
    "all_words"  : all_words,
    "tags"       : tags,
}

FILE = "data.pth"
torch.save(save_data, FILE)

print(f"Model saved to '{FILE}'")
print(f"Vocabulary : {len(all_words)} words")
print(f"Tags saved : {tags}")