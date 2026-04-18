import torch
import torch.nn as nn


class NeuralNet(nn.Module):
    """
    Feed-Forward Neural Network for intent classification.

    Architecture:
        Input Layer  -> Hidden Layer 1 -> Hidden Layer 2 -> Output Layer
        (bag size)       (hidden_size)      (hidden_size)     (num_tags)

    Activation: ReLU after each hidden layer.
    No activation on output — CrossEntropyLoss handles that internally.
    """

    def __init__(self, input_size, hidden_size, num_classes):
        """
        Args:
            input_size  : number of unique words in the vocabulary (bag of words size)
            hidden_size : number of neurons in each hidden layer (hyperparameter)
            num_classes : number of intent tags the model needs to classify
        """
        super(NeuralNet, self).__init__()

        # Layer 1: input -> hidden
        self.l1 = nn.Linear(input_size, hidden_size)

        # Layer 2: hidden -> hidden
        self.l2 = nn.Linear(hidden_size, hidden_size)

        # Layer 3: hidden -> output (one score per tag)
        self.l3 = nn.Linear(hidden_size, num_classes)

        # Activation function — applied after layers 1 and 2
        self.relu = nn.ReLU()

    def forward(self, x):
        """
        Forward pass: push input through all 3 layers.

        Args:
            x : input tensor (bag of words vector)

        Returns:
            out : raw scores for each tag (logits) — NOT probabilities yet
                  Use torch.softmax() on the output to get probabilities
        """
        out = self.relu(self.l1(x))   # input -> hidden 1 -> ReLU
        out = self.relu(self.l2(out)) # hidden 1 -> hidden 2 -> ReLU
        out = self.l3(out)            # hidden 2 -> output (no activation here)
        return out