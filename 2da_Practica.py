import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

# Función para particionar el dataset
def create_partitions(data, n_partitions, train_size=0.8):
    partitions = []
    for _ in range(n_partitions):
        train, test = train_test_split(data, test_size=1-train_size)
        partitions.append((train, test))
    return partitions

# Implementación del Perceptrón Simple
class Perceptron:
    def __init__(self, learning_rate=0.01, epochs=1000):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        # Convertir etiquetas a 1 o -1
        y_ = np.where(y <= 0, -1, 1)
        
        for _ in range(self.epochs):
            for idx, xi in enumerate(X):
                condition = y_[idx] * (np.dot(xi, self.weights) + self.bias) <= 0
                if condition:
                    update = self.learning_rate * y_[idx]
                    self.weights += update * xi
                    self.bias += update

    def predict(self, X):
        output = np.dot(X, self.weights) + self.bias
        return np.where(output <= 0, 0, 1)

# Leer el archivo y particionarlo
data = pd.read_csv('spheres1d10.csv')

# Suponiendo que tienes columnas 'x' para los datos y 'y' para las etiquetas
X = data.drop('y', axis=1).values
y = data['y'].values

partitions = create_partitions(data, 5, 0.8)

# Entrenar y evaluar el perceptrón en cada partición
for idx, (train, test) in enumerate(partitions):
    perceptron = Perceptron()
    X_train = train.drop('y', axis=1).values
    y_train = train['y'].values
    
    X_test = test.drop('y', axis=1).values
    y_test = test['y'].values
    
    perceptron.fit(X_train, y_train)
    predictions = perceptron.predict(X_test)
    accuracy = np.mean(predictions == y_test)
    
    print(f"Partición {idx + 1} - Exactitud: {accuracy * 100:.2f}%")

