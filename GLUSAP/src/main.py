from src import Simulation
from src.data_loader import OhioT1DDataLoader, ProportionalLSTMDataLoader
from src.models.model_lstm import LSTMModel
import pandas as pd
import numpy as np
import torch
import random
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)


adults = ['adult#001', 'adult#002', 'adult#003', 'adult#004', 'adult#005', 'adult#006', 'adult#007', 'adult#008',
        'adult#009', 'adult#010']

for patient in adults:
    sim = Simulation(days=10, patient=patient, sim_type='homogeneous', variability='low')
    sim.run_simulation()
    sim.generate_day_model()

    sim = Simulation(days=10, patient=patient, sim_type='homogeneous', variability='high')
    sim.run_simulation()
    sim.generate_day_model()

    sim = Simulation(days=10, patient=patient, sim_type='heterogeneous')
    sim.run_simulation()
    sim.generate_day_model()

filepaths = [
    '/Users/danasour/PycharmProjects/GLUSAP/data/processed/simglucose/heterogeneous_/adult#00*.csv',
    '/Users/danasour/PycharmProjects/GLUSAP/data/processed/simglucose/homogeneous_low/adult#00*.csv',
    '/Users/danasour/PycharmProjects/GLUSAP/data/processed/simglucose/homogeneous_high/adult#00*.csv'
]
categories = ["heterogeneous", "homogeneous_low", "homogeneous_high"]


loader = ProportionalLSTMDataLoader(filepaths, categories, seq_length=20, split_ratio=0.8)
data, data_pattern = loader.load_data()

sequences = loader.create_sequences(data)
X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor = loader.get_tensors()

print(f"Tamaño de X_train: {X_train_tensor.shape}")
print(f"Tamaño de y_train: {y_train_tensor.shape}")
print(f"Tamaño de X_test: {X_test_tensor.shape}")
print(f"Tamaño de y_test: {y_test_tensor.shape}")

model = LSTMModel()
model.train_model(X_train_tensor, y_train_tensor)

def generate_horizon_predictions(model, X_test_tensor, scaler, horizon=5):
    predictions = []

    for i in range(len(X_test_tensor)):
        current_input = X_test_tensor[i:i + 1]

        horizon_preds = []

        for j in range(horizon):
            pred = model.predict(current_input)
            horizon_preds.append(pred.item())

            current_input = np.roll(current_input, -1, axis=1)
            current_input[0, -1] = pred.item()  # Reemplazar el último valor con la predicción
            current_input = torch.tensor(current_input, dtype=torch.float32)

        predictions.append(horizon_preds)


    predictions = np.array(predictions)
    predictions_inv = scaler.inverse_transform(predictions.reshape(-1, 1)).reshape(predictions.shape)
    return predictions_inv

#####
patients = ['559', '563', '570', '575', '588', '591']

# Recórrer els pacients i calcular les mètriques de regressió per a cada un
for patient in patients:
    print(f"\nEvaluant el pacient {patient}:")

    data_loader = OhioT1DDataLoader(patient=patient)

    X_train_real_tensor, y_train_real_tensor = data_loader.get_train_tensors()
    X_test_real_tensor, y_test_real_tensor = data_loader.get_test_tensors()

    # Realitzar el fine-tuning amb les dades reals
    model.fine_tune(X_train_real_tensor, y_train_real_tensor)

    # Fer prediccions
    y_pred_real_test = model.predict(X_test_real_tensor)
    predictions_5_steps = generate_horizon_predictions(model, X_test_real_tensor, data_loader.scaler, horizon=5)

    # Desnormalitzar les prediccions i els valors reals
    y_pred_test_inv = data_loader.scaler.inverse_transform(y_pred_real_test.reshape(-1, 1))  # Remodelar a 2D
    y_test_inv = data_loader.scaler.inverse_transform(y_test_real_tensor.numpy().reshape(-1, 1))  # Remodelar a 2D

    metrics_by_horizon = {
        "Horizon": [],
        "MAE": [],
        "MSE": [],
        "R2": []
    }

    for horizon in range(5):
        real_values = y_test_inv[horizon:]
        pred_values = predictions_5_steps[:len(real_values), horizon]

        mae = mean_absolute_error(real_values, pred_values)
        mse = mean_squared_error(real_values, pred_values)
        r2 = r2_score(real_values, pred_values)

        metrics_by_horizon["Horizon"].append(horizon + 1)
        metrics_by_horizon["MAE"].append(mae)
        metrics_by_horizon["MSE"].append(mse)
        metrics_by_horizon["R2"].append(r2)

    metrics_df = pd.DataFrame(metrics_by_horizon)

    print(metrics_df)

    metrics_df.to_csv(f"data/metrics_by_horizon_patient_{patient}.csv", index=False)

    predictions_df = pd.DataFrame(predictions_5_steps, columns=[f"Pred_{i + 1}" for i in range(5)])

    test_times = data_loader.data['Time'][len(data_loader.data) - len(
        X_test_real_tensor):]  # Suponiendo que 'data' contiene las columnas de tiempo

    result_df = pd.DataFrame({
        'Time': test_times.to_numpy(),
        'Pred_1': predictions_df['Pred_1'],
        'Pred_2': predictions_df['Pred_2'],
        'Pred_3': predictions_df['Pred_3'],
        'Pred_4': predictions_df['Pred_4'],
        'Pred_5': predictions_df['Pred_5']
    })

    result_df.to_csv(f"data/predictions_patient_{patient}.csv", index=False)


    data_loader = OhioT1DDataLoader(patient=patient)
    data = data_loader.load_data()
    seq_length = 20
    train_size = int(len(data) * 0.8)

    y_pred_test_inv = y_pred_test_inv.flatten()

    data.iloc[seq_length:train_size + seq_length].to_csv(f"data/train_data_{patient}.csv", index=False)
    data.iloc[train_size + seq_length:].to_csv(f"data/test_data_{patient}.csv", index=False)
