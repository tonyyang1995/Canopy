from sklearn.metrics import r2_score
from sklearn import linear_model
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
import xgboost as xgb
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.metrics import r2_score

def calculate_rmse(actual, predicted):
    # Ensure the inputs are numpy arrays
    actual = np.array(actual)
    predicted = np.array(predicted)
    
    # Calculate the RMSE
    rmse = np.sqrt(np.mean((actual - predicted) ** 2))
    return rmse

def calculate_rmae(actual, predicted):
    # Ensure the inputs are numpy arrays
    actual = np.array(actual)
    predicted = np.array(predicted)
    
    # Calculate the RMAE
    rmae = np.sqrt(np.mean(np.abs(actual - predicted)))
    return rmae

def calculate_r2(actual, predicted):
    # Ensure the inputs are numpy arrays
    actual = np.array(actual)
    predicted = np.array(predicted)
    
    # Calculate the total sum of squares (SST) and residual sum of squares (SSR)
    ss_total = np.sum((actual - np.mean(actual)) ** 2)
    ss_residual = np.sum((actual - predicted) ** 2)
    
    # Calculate R2
    r2 = 1 - (ss_residual / ss_total)
    return r2

def calculate_adjusted_r2(actual, predicted, num_predictors=13):
    # Ensure the inputs are numpy arrays
    actual = np.array(actual)
    predicted = np.array(predicted)
    
    # Calculate the total sum of squares (SST) and residual sum of squares (SSR)
    ss_total = np.sum((actual - np.mean(actual)) ** 2)
    ss_residual = np.sum((actual - predicted) ** 2)
    
    # Calculate R2
    r2 = 1 - (ss_residual / ss_total)
    
    # Calculate Adjusted R2
    n = len(actual)  # Number of observations
    adjusted_r2 = 1 - ((1 - r2) * (n - 1)) / (n - num_predictors - 1)
    
    return adjusted_r2

def calculate_RSR(actual, prediction):
    rmse = calculate_rmae(actual, prediction)
    diff_sum = (actual - prediction) ** 2
    diff_sum = np.sum(diff_sum)
    scale = np.sqrt(diff_sum)
    return rmse / scale

def calculate_mape(actual, prediction):
    actual = np.array(actual)
    prediction = np.array(prediction)
    return np.mean(np.abs((actual - prediction) / actual)) * 100

def calcuate_nmbe(actual, prediction):
    actual = np.array(actual)
    prediction = np.array(prediction)
    return np.sum(actual - prediction) / np.sum(actual) * 100



class RidgeRegressionModel():
    def __init__(self, inputs):
        self.inputs = inputs
        self.model = linear_model.RidgeCV(alphas=np.logspace(-6, 6, 13))
        self.RMSE = []
        self.RMAE = []
        self.R2 = []
        self.RSR = []
        self.MAPE = []
        self.NMBE = []
        self.predictions = []

    def fit(self):
        for x_train, y_train, x_test, y_test in self.inputs:
            self.model.fit(x_train, y_train)
            predictions = self.model.predict(x_test)
            self.RMSE.append(calculate_rmse(y_test, predictions))
            self.RMAE.append(calculate_rmae(y_test, predictions))
            self.R2.append(r2_score(y_test, predictions))
            self.RSR.append(calculate_RSR(y_test, predictions))
            self.MAPE.append(calculate_mape(y_test, predictions))
            self.NMBE.append(calcuate_nmbe(y_test, predictions))
            self.predictions.append([y_test, predictions])

    def get_results(self):
        return {
            "RMSE": self.RMSE,
            "RMAE": self.RMAE,
            "R2": self.R2,
            "RSR": self.RSR,
            "MAPE": self.MAPE,
            "NMBE": self.NMBE,
        }

    def get_predictions(self):
        return self.predictions

class RandomForestModel():
    def __init__(self, inputs):
        self.inputs = inputs
        self.model = RandomForestRegressor(random_state=42)
        self.RMSE = []
        self.RMAE = []
        self.R2 = []
        self.RSR = []
        self.MAPE = []
        self.NMBE = []
        self.predictions = []

    def fit(self):
        for x_train, y_train, x_test, y_test in self.inputs:
            self.model.fit(x_train, y_train)
            predictions = self.model.predict(x_test)
            self.RMSE.append(calculate_rmse(y_test, predictions))
            self.RMAE.append(calculate_rmae(y_test, predictions))
            self.R2.append(r2_score(y_test, predictions))
            self.RSR.append(calculate_RSR(y_test, predictions))
            self.MAPE.append(calculate_mape(y_test, predictions))
            self.NMBE.append(calcuate_nmbe(y_test, predictions))
            self.predictions.append([y_test, predictions])

    def get_results(self):
        return {
            "RMSE": self.RMSE,
            "RMAE": self.RMAE,
            "R2": self.R2,
            "RSR": self.RSR,
            "MAPE": self.MAPE,
            "NMBE": self.NMBE,
        }

    def get_predictions(self):
        return self.predictions

class SVMRegressor():
    def __init__(self, inputs):
        self.inputs = inputs
        self.model = SVR()
        self.RMSE = []
        self.RMAE = []
        self.R2 = []
        self.RSR = []
        self.MAPE = []
        self.NMBE = []
        self.predictions = []

    def fit(self):
        for x_train, y_train, x_test, y_test in self.inputs:
            self.model.fit(x_train, y_train)
            predictions = self.model.predict(x_test)
            self.RMSE.append(calculate_rmse(y_test, predictions))
            self.RMAE.append(calculate_rmae(y_test, predictions))
            self.R2.append(r2_score(y_test, predictions))
            self.RSR.append(calculate_RSR(y_test, predictions))
            self.MAPE.append(calculate_mape(y_test, predictions))
            self.NMBE.append(calcuate_nmbe(y_test, predictions))
            self.predictions.append([y_test, predictions])


    def get_results(self):
        return {
            "RMSE": self.RMSE,
            "RMAE": self.RMAE,
            "R2": self.R2,
            "RSR": self.RSR,
            "MAPE": self.MAPE,
            "NMBE": self.NMBE,
        }
    
    def get_predictions(self):
        return self.predictions

class XGBregressor():
    def __init__(self, inputs):
        self.inputs = inputs
        self.model = xgb.XGBRegressor(n_estimators=1000)
        self.RMSE = []
        self.RMAE = []
        self.R2 = []
        self.RSR = []
        self.MAPE = []
        self.NMBE = []
        self.predictions = []

    def fit(self):
        for x_train, y_train, x_test, y_test in self.inputs:
            self.model.fit(x_train, y_train)
            predictions = self.model.predict(x_test)
            self.RMSE.append(calculate_rmse(y_test, predictions))
            self.RMAE.append(calculate_rmae(y_test, predictions))
            self.R2.append(r2_score(y_test, predictions))
            self.RSR.append(calculate_RSR(y_test, predictions))
            self.MAPE.append(calculate_mape(y_test, predictions))
            self.NMBE.append(calcuate_nmbe(y_test, predictions))
            self.predictions.append([y_test, predictions])


    def get_results(self):
        return {
            "RMSE": self.RMSE,
            "RMAE": self.RMAE,
            "R2": self.R2,
            "RSR": self.RSR,
            "MAPE": self.MAPE,
            "NMBE": self.NMBE,
        }
    
    def get_predictions(self):
        return self.predictions

class MLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        """
        input_size: Number of input features
        hidden_sizes: List containing the sizes of each hidden layer
        output_size: Number of output features
        """
        super(MLP, self).__init__()
        
        # Define the first layer from input to first hidden layer
        layers = [nn.Linear(input_size, hidden_sizes[0])]
        
        # Create the hidden layers dynamically based on hidden_sizes
        for i in range(1, len(hidden_sizes)):
            layers.append(nn.Linear(hidden_sizes[i-1], hidden_sizes[i]))
        
        # Append the final layer that maps the last hidden layer to the output
        layers.append(nn.Linear(hidden_sizes[-1], output_size))
        
        # Store the layers in an nn.ModuleList
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        # Pass the input through each layer, applying ReLU after each layer except the output
        for i in range(len(self.layers) - 1):
            x = F.relu(self.layers[i](x))
        
        # Output layer (without activation, useful for regression)
        x = self.layers[-1](x)
        
        return x

class ANNregressor():
    def __init__(self, inputs):
        self.inputs = inputs
        self.models = []
        for input in inputs:
            self.models.append(MLP(input_size=input[0].shape[1], hidden_sizes=[32, 128, 128], output_size=1))
        self.RMSE = []
        self.RMAE = []
        self.R2 = []
        self.RSR = []
        self.MAPE = []
        self.NMBE = []
        self.criterion = nn.MSELoss()
        self.predictions = []

    def train_a_model(self, model, x_train, y_train, x_test, y_test):
        model.train()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        for epoch in range(1000):
            output = model(torch.from_numpy(x_train.astype(np.float32)))
            loss = self.criterion(output, torch.from_numpy(y_train.astype(np.float32)).unsqueeze(1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        y_pred = model(torch.from_numpy(x_test.astype(np.float32))).detach().squeeze(1).numpy()
        rmse = calculate_rmse(y_test, y_pred)
        rmae = calculate_rmae(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        rsr = calculate_RSR(y_test, y_pred)
        mape = calculate_mape(y_test, y_pred)
        nmbe = calcuate_nmbe(y_test, y_pred)
        return rmse, rmae, r2, y_pred, rsr, mape, nmbe

    def fit(self):
        for i in range(len(self.models)):
            x_train, y_train, x_test, y_test = self.inputs[i]
            rmse, rmae, r2, y_pred, rsr, mape, nmbe = self.train_a_model(self.models[0], x_train, y_train, x_test, y_test)

            self.RMSE.append(rmse)
            self.RMAE.append(rmae)
            self.R2.append(r2)
            self.predictions.append([y_test, y_pred])
            self.RSR.append(rsr)
            self.MAPE.append(mape)
            self.NMBE.append(nmbe)

    def get_results(self):
        return {
            "RMSE": self.RMSE,
            "RMAE": self.RMAE,
            "R2": self.R2,
            "RSR": self.RSR,
            "MAPE": self.MAPE,
            "NMBE": self.NMBE,
        }

    def get_predictions(self):
        return self.predictions