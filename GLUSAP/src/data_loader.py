"""Script containing the DataLoader class to load and preprocess the data."""
import xml.etree.ElementTree as ET
import glob
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import torch

class DataLoaderBase:
    def __init__(self, filepath, seq_length=20, split_ratio=0.8, resample_freq='15T'):
        self.filepath = filepath
        self.seq_length = seq_length
        self.split_ratio = split_ratio
        self.resample_freq = resample_freq
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.targets_normalized = None
        self.targets = None
        self.train_data = None
        self.test_data = None

    def load_data(self):
        data = pd.read_csv(self.filepath)
        data['Time'] = pd.to_datetime(data['Time'])
        data.set_index('Time', inplace=True)

        self.targets = data['CGM'].values
        self.targets_normalized = self.scaler.fit_transform(self.targets.reshape(-1, 1)).flatten()

        return data

    def resample_data(self, data):
        data_resampled = data.resample(self.resample_freq).mean().interpolate()

        # Set index
        data_resampled.index = pd.date_range(start=data_resampled.index[0],
                                             periods=len(data_resampled), freq=self.resample_freq)
        return data_resampled

    def split_data(self):
        train_size = int(len(self.targets_normalized) * self.split_ratio)
        self.train_data = self.targets_normalized[:train_size]
        self.test_data = self.targets_normalized[train_size:]

        return self.train_data, self.test_data


class LSTMDataLoader(DataLoaderBase):
    def __init__(self, filepath, seq_length=20, split_ratio=0.8):
        super().__init__(filepath, seq_length, split_ratio)

    def create_sequences(self):
        xs, ys = [], []
        for i in range(len(self.targets_normalized) - self.seq_length):
            x = self.targets_normalized[i:i + self.seq_length]
            y = self.targets_normalized[i + self.seq_length]
            xs.append(x)
            ys.append(y)
        X, y = np.array(xs), np.array(ys)

        train_size = int(len(X) * self.split_ratio)
        self.X_train, self.X_test = X[:train_size], X[train_size:]
        self.y_train, self.y_test = y[:train_size], y[train_size:]

        return self.X_train, self.X_test, self.y_train, self.y_test

    def get_tensors(self):
        X_train_tensor = torch.FloatTensor(self.X_train).view(-1, self.seq_length, 1)
        y_train_tensor = torch.FloatTensor(self.y_train)
        X_test_tensor = torch.FloatTensor(self.X_test).view(-1, self.seq_length, 1)
        y_test_tensor = torch.FloatTensor(self.y_test)

        return X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data.reshape(-1, 1))


class SARIMAXDataLoader(DataLoaderBase):
    def __init__(self, filepath, seq_length=20, split_ratio=0.8):
        super().__init__(filepath, seq_length, split_ratio)

    def prepare_data(self, small_data=True):
        self.load_data()

        if small_data:
            self.train_data = self.targets_normalized[-8 * 96:]
            self.test_data = self.targets_normalized[:3 * 96]

        train_data_grouped = self.resample_data(pd.Series(self.train_data))
        test_data_grouped = self.resample_data(pd.Series(self.test_data))

        return train_data_grouped, test_data_grouped


class OhioT1DDataLoader(DataLoaderBase):
    def __init__(self, patient, filepath='data/raw/OhioT1DM_2018', seq_length=20, split_ratio=0.8):
        super().__init__(patient, seq_length, split_ratio)
        self.patient = patient
        self.processed_filepath = f'data/processed/'
        self.filename = f'{self.processed_filepath}OhioT1DM_{patient}.csv'
        self.filepaths = [f"{filepath}/{patient}-ws-training.xml",
                         f"{filepath}/{patient}-ws-testing.xml"]
        self.data = self.load_data()
        self.train_data, self.test_data = self.split_data()

        self.X_train, self.y_train = self.create_sequences(self.train_data)
        self.X_test, self.y_test = self.create_sequences(self.test_data)

    def get_data(self):
        return self.train_data, self.test_data
    def load_data(self):
        try:
            data = pd.read_csv(self.filename)
            self.targets = data['CGM'].values
            self.targets_normalized = self.scaler.fit_transform(self.targets.reshape(-1, 1)).flatten()

            df_reindexed = data.set_index('Time')
            df_reindexed.index = pd.date_range(start='2024-01-01', periods=len(df_reindexed), freq='5T')
            df_reindexed.reset_index(inplace=True)
            df_reindexed.rename(columns={'index': 'Time'}, inplace=True)
            df_reindexed.to_csv(self.processed_filepath + f'OhioT1DM_{self.patient}_processed.csv')

            return data
        except FileNotFoundError:
            data = []
            for file in self.filepaths:
                tree = ET.parse(file)
                root = tree.getroot()

                for event in root.find('glucose_level'):
                    timestamp = event.get('ts')
                    value = float(event.get('value'))
                    data.append({'Time': timestamp, 'CGM': value})

            df = pd.DataFrame(data)
            df['Time'] = pd.to_datetime(df['Time'], format='%d-%m-%Y %H:%M:%S')
            df.set_index('Time', inplace=True)

            df.to_csv(self.filename)

            self.targets = df['CGM'].values
            self.targets_normalized = self.scaler.fit_transform(self.targets.reshape(-1, 1)).flatten()

        return df

    def create_sequences(self, data):
        sequences, targets = [], []
        for i in range(len(data) - self.seq_length):
            sequences.append(data[i:i + self.seq_length])
            targets.append(data[i + self.seq_length])
        return np.array(sequences), np.array(targets)

    def get_train_tensors(self):
        X_train_tensor = torch.FloatTensor(self.X_train).unsqueeze(-1)
        y_train_tensor = torch.FloatTensor(self.y_train)
        return X_train_tensor, y_train_tensor

    def get_test_tensors(self):
        X_test_tensor = torch.FloatTensor(self.X_test).unsqueeze(-1)
        y_test_tensor = torch.FloatTensor(self.y_test)
        return X_test_tensor, y_test_tensor

class ProportionalLSTMDataLoader:
    def __init__(self, filepaths, categories, seq_length=20, split_ratio=0.8):
        """
        Args:
            filepaths (list): Lista de patrones de archivos para cada categoría.
            categories (list): Lista de categorías correspondientes.
            seq_length (int): Longitud de las secuencias.
            split_ratio (float): Proporción de división entre entrenamiento y prueba.
        """
        self.filepaths = filepaths
        self.categories = categories
        self.seq_length = seq_length
        self.split_ratio = split_ratio

        self.data = {category: [] for category in categories}
        self.data_withcho = {category: [] for category in categories}

        self.load_data()
        self.train, self.test = self.split_data()

        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.X_train, self.y_train = self.create_sequences(self.train)
        self.X_test, self.y_test = self.create_sequences(self.test)

    def load_data(self):
        for filepath, category in zip(self.filepaths, self.categories):
            for file in glob.glob(filepath):
                if "day_model" in file:
                    continue
                dataset = pd.read_csv(file)
                self.data[category].append(dataset['CGM'].values)
                self.data_withcho[category].append(dataset[['Time','CGM', 'CHO']])
        return self.data, self.data_withcho

    def split_data(self):
        """
        Returns:
            train (dict): Datos de entrenamiento por categoría.
            test (dict): Datos de prueba por categoría.
        """
        train = {}
        test = {}
        for key, arrays in self.data.items():
            indices = np.arange(len(arrays))
            np.random.shuffle(indices)  # Barajar índices
            split_point = int(len(arrays) * self.split_ratio)
            train_indices = indices[:split_point]
            test_indices = indices[split_point:]
            train[key] = [arrays[i] for i in train_indices]
            test[key] = [arrays[i] for i in test_indices]
        return train, test

    def normalize_data(self, array):
        """
        Args:
            array (np.array): Array de datos a normalizar.

        Returns:
            np.array: Array normalizado.
        """
        return self.scaler.fit_transform(array.reshape(-1, 1)).flatten()

    def create_sequences(self, data_dict):
        """
        Args:
            data_dict (dict): Datos organizados por categoría.

        Returns:
            X (list): Secuencias de entrada.
            y (list): Etiquetas correspondientes.
        """
        X, y = [], []
        for key, arrays in data_dict.items():
            for array in arrays:
                normalized_array = self.normalize_data(array)
                for i in range(len(normalized_array) - self.seq_length):
                    X.append(normalized_array[i:i + self.seq_length])
                    y.append(normalized_array[i + self.seq_length])
        return np.array(X), np.array(y)

    def get_tensors(self):
        """

        Returns:
            X_train_tensor (torch.Tensor): Tensor de entradas de entrenamiento.
            y_train_tensor (torch.Tensor): Tensor de etiquetas de entrenamiento.
            X_test_tensor (torch.Tensor): Tensor de entradas de prueba.
            y_test_tensor (torch.Tensor): Tensor de etiquetas de prueba.
        """
        X_train_tensor = torch.FloatTensor(self.X_train).view(-1, self.seq_length, 1)
        y_train_tensor = torch.FloatTensor(self.y_train)
        X_test_tensor = torch.FloatTensor(self.X_test).view(-1, self.seq_length, 1)
        y_test_tensor = torch.FloatTensor(self.y_test)
        return X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor

    def inverse_transform(self, data):
        """
        Args:
            data (np.array): Datos normalizados a desnormalizar.

        Returns:
            np.array: Datos desnormalizados.
        """
        return self.scaler.inverse_transform(data.reshape(-1, 1))



