# Clase para el modelo LSTM
import numpy as np
import torch.nn as nn
import torch
from matplotlib import pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, f1_score


class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, num_layers=2, dropout=0.2):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

    def train_model(self, X_train, y_train, epochs=60, lr=0.01):
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        for epoch in range(epochs):
            self.train()
            optimizer.zero_grad()
            output = self(X_train)
            loss = criterion(output.squeeze(), y_train)
            loss.backward()
            optimizer.step()
            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')

    def evaluate_regression(self, X_test, y_test):
        """
        Funció per avaluar el model amb estadístiques de regressió.
        """
        self.eval()
        with torch.no_grad():
            y_pred = self.predict(X_test)

        # Càlcul dels errors
        mae = mean_absolute_error(y_test.numpy(), y_pred)
        mse = mean_squared_error(y_test.numpy(), y_pred)
        r2 = r2_score(y_test.numpy(), y_pred)

        print(f"Mean Absolute Error (MAE): {mae:.4f}")
        print(f"Mean Squared Error (MSE): {mse:.4f}")
        print(f"R² Score: {r2:.4f}")

        return mae, mse, r2

    def evaluate_f1_with_strict_intervals(self, X_test, y_test, confidence_intervals=[0.05, 0.10, 0.15]):
        """
        Funció per avaluar el model amb F1 Score utilitzant intervals de confiança més estrictes.

        Paràmetres:
        - confidence_intervals (list): Llista de percentatges per a definir els intervals de confiança al voltant de la predicció.
        """
        # Predicció
        self.eval()
        with torch.no_grad():
            y_pred = self.predict(X_test)

        # Inicialitzar resultats
        results = {}

        # Per cada interval de confiança especificat
        for ci in confidence_intervals:
            lower_bound = y_pred * (1 - ci / 100)
            upper_bound = y_pred * (1 + ci / 100)

            # Convertir les etiquetes i prediccions a categories de "Sí" o "No"
            y_pred_class = [
                (real >= lb and real <= ub and abs(real - pred) / pred < ci / 100)
                for lb, ub, real, pred in zip(lower_bound, upper_bound, y_test.numpy(), y_pred)
            ]
            y_test_class = [True] * len(y_test)  # Assume que totes les etiquetes reals són vàlides

            # Càlcul de l'F1 score
            f1 = f1_score(y_test_class, y_pred_class)

            # Calcular VP, FP, FN, VN
            VP = np.sum(np.array(y_test_class) & np.array(y_pred_class))
            FP = np.sum(np.array(y_pred_class) & ~np.array(y_test_class))
            FN = np.sum(~np.array(y_pred_class) & np.array(y_test_class))
            VN = np.sum(~np.array(y_test_class) & ~np.array(y_pred_class))

            # Almacenar els resultats
            results[ci] = {
                "F1 Score": f1,
                "VP": VP,
                "FP": FP,
                "FN": FN,
                "VN": VN,
                "Accuracy": (VP + VN) / (VP + FP + FN + VN),  # Exactitud general
            }

        # Mostrar els resultats
        for ci, stats in results.items():
            print(f"Interval de Confiança: ±{ci}%")
            print(f"  F1 Score: {stats['F1 Score']:.4f}")
            print(f"  VP: {stats['VP']}")
            print(f"  FP: {stats['FP']}")
            print(f"  FN: {stats['FN']}")
            print(f"  VN: {stats['VN']}")
            print(f"  Accuracy: {stats['Accuracy']:.4f}")
            print("-" * 50)

        return results

    def fine_tune(self, X_train, y_train, epochs=5, lr=0.01):
        for param in self.lstm.parameters():
            param.requires_grad = False  # Congela las capas LSTM

        for param in self.fc.parameters():
            param.requires_grad = True

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=lr)

        for epoch in range(epochs):
            self.train()
            optimizer.zero_grad()
            y_pred = self(X_train)
            loss = criterion(y_pred.squeeze(), y_train)
            loss.backward()
            optimizer.step()

            print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')

    def predict(self, X_tensor):
        self.eval()
        with torch.no_grad():
            return self(X_tensor).numpy()

    def dynamic_prediction_with_real_data_hourly(self, X_train_tensor, y_test_tensor, period_duration=20):
        self.eval()
        dynamic_predictions = []
        test_input = X_train_tensor[-1].unsqueeze(0)

        for i in range(0, len(y_test_tensor), period_duration):
            hour_predictions = []
            for j in range(period_duration):
                with torch.no_grad():
                    prediction = self(test_input).squeeze()
                    hour_predictions.append(prediction.item())

                if i + j + 1 < len(y_test_tensor):
                    next_real_value = y_test_tensor[i + j].item()
                    test_input = torch.cat((test_input[:, 1:, :], torch.FloatTensor([[[next_real_value]]])), dim=1)

            dynamic_predictions.extend(hour_predictions)
        return np.array(dynamic_predictions)

