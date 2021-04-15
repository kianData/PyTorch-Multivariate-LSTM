# -*- coding: utf-8 -*-
"""
Created on Sun Oct 25 11:17:43 2020

@author: Kianoosh Keshavarzian
"""

import numpy as np
from numpy import array
from numpy import hstack
import pandas as pd
import matplotlib.pyplot as plt
import math
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import torch
import torch.nn as nn

# split a multivariate sequence into samples
def split_sequences(sequences, n_steps_in, n_steps_out):
	X, y = list(), list()
	for i in range(len(sequences)):
		# find the end of this pattern
		end_ix = i + n_steps_in
		out_end_ix = end_ix + n_steps_out-1
		# check if we are beyond the dataset
		if out_end_ix > len(sequences):
			break
		# gather input and output parts of the pattern
		seq_x, seq_y = sequences[i:end_ix, :-1], sequences[end_ix-1:out_end_ix, -1]
		X.append(seq_x)
		y.append(seq_y)
	return array(X), array(y)

df = pd.read_csv('goldETF.csv')

# define input sequence
in_seq1 = array(df['High'].values)
in_seq2 = array(df['Low'].values)
in_seq3 = array(df['Open'].values)
in_seq4 = array(df['Close'].values)
in_seq5 = array(df['Volume'].values)

out_seq = array(df['Close'].values)

# convert to [rows, columns] structure
in_seq1 = in_seq1.reshape((len(in_seq1), 1))
in_seq2 = in_seq2.reshape((len(in_seq2), 1))
in_seq3 = in_seq3.reshape((len(in_seq3), 1))
in_seq4 = in_seq4.reshape((len(in_seq4), 1))
in_seq5 = in_seq5.reshape((len(in_seq5), 1))
out_seq = out_seq.reshape((len(out_seq), 1))

# horizontally stack columns
dataset = hstack((in_seq1, in_seq2, in_seq3, in_seq4, in_seq5, out_seq))

# scaling dataset ===============================================
in_seq = np.concatenate((in_seq1, in_seq2, in_seq3, in_seq4, out_seq), axis=0)
scaler = MinMaxScaler(feature_range=(0, 1))
scaler_all = scaler.fit(in_seq)

scaled_in_seq1 = scaler_all.transform(in_seq1)
scaled_in_seq2 = scaler_all.transform(in_seq2)
scaled_in_seq3 = scaler_all.transform(in_seq3)
scaled_in_seq4 = scaler_all.transform(in_seq4)
scaled_in_seq5 = scaler_all.transform(in_seq5)
scaled_out_seq = scaler_all.transform(out_seq)

scaled_data = hstack((scaled_in_seq1, scaled_in_seq2, scaled_in_seq3, scaled_in_seq4, scaled_out_seq))

# choose a number of time steps
n_steps_in, n_steps_out = 60, 1

# convert into input/output
x_train, y_train = split_sequences(scaled_data, n_steps_in, n_steps_out)

train_set_size = int(0.2*scaled_data.shape[0])
x_test, y_test = split_sequences(scaled_data[-train_set_size:-1,:], n_steps_in, n_steps_out)

# make training and test sets in torch
x_train = torch.from_numpy(x_train).type(torch.Tensor)
x_test = torch.from_numpy(x_test).type(torch.Tensor)
y_train = torch.from_numpy(y_train).type(torch.Tensor)
y_test = torch.from_numpy(y_test).type(torch.Tensor)

y_train.size(),x_train.size()

num_epochs = 50

# Build model
##################################################

input_dim = 4
hidden_dim = 32
num_layers = 2
output_dim = 1

# Here we define our model as a class
class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(LSTM, self).__init__()
        # Hidden dimensions
        self.hidden_dim = hidden_dim

        # Number of hidden layers
        self.num_layers = num_layers

        # Building your LSTM
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)

        # Readout layer
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()

        # Initialize cell state
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()

        # One time step
        # We need to detach as we are doing truncated backpropagation through time (BPTT)
        # If we don't, we'll backprop all the way to the start even after going through another batch
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))

        # Index hidden state of last time step
        out = self.fc(out[:, -1, :]) 
        
        return out
    
model = LSTM(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, num_layers=num_layers)

loss_fn = torch.nn.MSELoss(size_average=True)

optimiser = torch.optim.Adam(model.parameters(), lr=0.01)
print(model)
print(len(list(model.parameters())))
for i in range(len(list(model.parameters()))):
    print(list(model.parameters())[i].size())
    
# Train model
##################################################################

hist = np.zeros(num_epochs)

for t in range(num_epochs):
    # Initialise hidden state
        
    # Forward pass
    y_train_pred = model(x_train)

    loss = loss_fn(y_train_pred, y_train)
    if t % 10 == 0 and t !=0:
        print("Epoch ", t, "MSE: ", loss.item())
    hist[t] = loss.item()

    # Zero out gradient, else they will accumulate between epochs
    optimiser.zero_grad()

    # Backward pass
    loss.backward()

    # Update parameters
    optimiser.step()

plt.plot(y_train_pred.detach().numpy(), label="Preds")
plt.plot(y_train.detach().numpy(), label="Data")
plt.legend()
plt.show()

plt.plot(hist, label="Training loss")
plt.legend()
plt.show()

# make predictions
y_test_pred = model(x_test)

# invert predictions
y_train_pred = scaler_all.inverse_transform(y_train_pred.detach().numpy())
y_train = scaler_all.inverse_transform(y_train.detach().numpy())
y_test_pred = scaler_all.inverse_transform(y_test_pred.detach().numpy())
y_test = scaler_all.inverse_transform(y_test.detach().numpy())

# calculate root mean squared error
trainScore = math.sqrt(mean_squared_error(y_train[:,0], y_train_pred[:,0]))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(y_test[:,0], y_test_pred[:,0]))
print('Test Score: %.2f RMSE' % (testScore))

# plot baseline and predictions
plt.figure(figsize=(15,8))
plt.plot(y_train_pred)
plt.plot(y_train)
plt.show()

# plot baseline and predictions
plt.figure(figsize=(15,8))
plt.plot(y_test_pred)
plt.plot(y_test)
plt.show()