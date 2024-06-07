import helpermlp
import numpy as np
import flwr as fl
from sklearn.metrics import log_loss, accuracy_score, precision_score, recall_score, f1_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
import warnings
warnings.simplefilter('ignore')

class FlowerClient(fl.client.NumPyClient):

    def get_parameters(self, config):
        return helpermlp.get_params(model)
    
    def fit(self, parameters, config):
        helpermlp.set_params(model, parameters)
        model.fit(X_train_scaled, y_train)
        return helpermlp.get_params(model), len(X_train), {}

    def evaluate(self, parameters, config):
        helpermlp.set_params(model, parameters)
        y_pred = model.predict(X_test_scaled)
        loss = log_loss(y_test, y_pred)

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')

        return loss, len(X_test), {"Accuracy": accuracy, "Precision": precision, "Recall": recall, "F1_Score": f1}

if __name__ == "__main__":
    client_id = 1
    print(f"Client {client_id}:\n")

    X_train, y_train, X_test, y_test = helpermlp.load_dataset(client_id - 1)

    unique, counts = np.unique(y_train, return_counts=True)
    train_counts = dict(zip(unique, counts))
    print("Label distribution in the training set:", train_counts)
    unique, counts = np.unique(y_test, return_counts=True)
    test_counts = dict(zip(unique, counts))
    print("Label distribution in the testing set:", test_counts, '\n')

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = MLPClassifier(hidden_layer_sizes=(200,), max_iter=200, alpha=1e-4,
                          solver='adam', random_state=1, learning_rate_init=0.001)

    model.fit(X_train_scaled, y_train)
    fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=FlowerClient())

