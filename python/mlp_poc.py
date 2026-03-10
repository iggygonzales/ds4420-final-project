""" POC for a manual MLP regression model to predict renewable electricity share from economic/energy features"""
import pandas as pd
import numpy as np

# Load & clean data
df = pd.read_csv("data/World_Energy_Consumption.csv")

# Filter to a single year so each row = one country (one observation)
# iso_code filters out regional aggregates like "Asia (Ember)" that would mess up the model
df2020 = df[(df["year"] == 2020) & df["iso_code"].notna()]

# These are our input features purely economic/energy profile signals
# We deliberately exclude any renewable-derived columns (e.g. solar_share_elec) so the model learns from structural features, not circular ones
FEATURES = ["population", "energy_per_capita", "fossil_share_elec",
            "coal_share_elec", "gas_share_elec", "per_capita_electricity"]
TARGET = "renewables_share_elec"

# Drop any row missing even one value, the MLP can't handle NaNs
df_clean = df2020[["country"] + FEATURES + [TARGET]].dropna()
print(f"Countries: {len(df_clean)}")  


# Forward pass 
# We convert to numpy arrays and normalise so the MLP trains better
X = df_clean[FEATURES].values.astype(float)        # shape: (N, 6)
y = df_clean[TARGET].values.astype(float).reshape(-1, 1)  # shape: (N, 1)

X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)  # normalise
y = (y - y.min()) / (y.max() - y.min())             # normalise target to [0,1] too

# Small hidden layer (8 neurons) just to verify shapes work end-to-end
# Full model will use 64 > 32, but that doesn't matter for the POC
# * 0.01 keeps initial weights small so activations don't saturate immediately
W1 = np.random.randn(X.shape[1], 8) * 0.01  
b1 = np.zeros((1, 8))                        
W2 = np.random.randn(8, 1) * 0.01           
b2 = np.zeros((1, 1))                        

# @ is matrix multiplication, this is the core of the forward pass
# each neuron computes a weighted sum of its inputs, then adds a bias
Z1 = X @ W1 + b1      # pre-activation: (N, 8)
A1 = np.maximum(0, Z1) # ReLU: kills negative values, passes positive ones through
                        # this is what lets the network learn non-linear patterns
Z2 = A1 @ W2 + b2     # output layer: (N, 1)  no activation, raw regression output

print(Z2.shape)        # must be (N, 1) one prediction per country


# Training loop

# Learning rate controls how big a step we take in the direction of the negative gradient each update
lr = 0.001

for epoch in range(20):

    # Forward pass 
    Z1 = X @ W1 + b1 # pre-activation: (N, 8)
    A1 = np.maximum(0, Z1) # ReLU: kills negative values, passes positive ones through
    Z2 = A1 @ W2 + b2 # output layer: (N, 1) no activation, raw regression output

    # Mean squared error: average squared difference between prediction and truth
    # This is what we're minimising, the gradients below are its derivatives
    loss = np.mean((Z2 - y) ** 2)


    # Backward pass (chain rule, layer by layer) 

    # Derivative of MSE loss w.r.t. Z2 (output layer pre-activation)
    # The /len(y) averages the gradient across all N samples
    dZ2 = 2 * (Z2 - y) / len(y)       # (N, 1)

    # How much each weight in W2 contributed to the loss
    # A1.T transposes so shapes align: (8, N) @ (N, 1) = (8, 1)
    dW2 = A1.T @ dZ2                   # (8, 1)

    # Propagate gradient back through W2 into hidden layer
    # Then apply ReLU derivative: gradient is zero wherever Z1 was negative
    # (Z1 > 0) is a binary mask - same idea as the forward ReLU
    dZ1 = (dZ2 @ W2.T) * (Z1 > 0)    # (N, 8)

    # How much each weight in W1 contributed to the loss
    dW1 = X.T @ dZ1                   # (6, N) @ (N, 8) = (6, 8)

    # Gradient descent update 
    W1 -= lr * dW1 # update W1 by stepping in the direction of the negative gradient
    W2 -= lr * dW2 # update W2 by stepping in the direction of the negative gradient
    b1 -= lr * dZ1.sum(axis=0, keepdims=True)  # bias gradients are the sum of dZ1 across all samples
    b2 -= lr * dZ2.sum(axis=0, keepdims=True)  # bias gradients are the sum of dZ2 across all samples

    print(f"Epoch {epoch+1}: loss = {loss:.4f}")  # should decrease each line