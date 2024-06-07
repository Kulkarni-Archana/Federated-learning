import flwr as fl
from typing import Dict
import matplotlib.pyplot as plt

ROUNDS = 3
evaluate_accuracies = []

def fit_config(server_round: int) -> Dict:
    config = {
        "server_round": server_round,
    }
    return config

def evaluate_metrics_aggregate(results) -> Dict:
    if not results:
        return {}

    total_samples = 0
    aggregated_metrics = {
        "Accuracy": 0,
        "Precision": 0,
        "Recall": 0,
        "F1_Score": 0,
    }

    for samples, metrics in results:
        for key, value in metrics.items():
            if key not in aggregated_metrics:
                aggregated_metrics[key] = 0
            else:
                aggregated_metrics[key] += (value * samples)
        total_samples += samples

    for key in aggregated_metrics.keys():
        aggregated_metrics[key] = round(aggregated_metrics[key] / total_samples, 6)

    evaluate_accuracies.append(aggregated_metrics["Accuracy"])
    return aggregated_metrics

def fit_metrics_aggregate(results) -> Dict:
    if not results:
        return {}

    total_samples = 0
    aggregated_metrics = {
        "Accuracy": 0,
        "Precision": 0,
        "Recall": 0,
        "F1_Score": 0,
    }

    for samples, metrics in results:
        for key, value in metrics.items():
            if key not in aggregated_metrics:
                aggregated_metrics[key] = 0
            else:
                aggregated_metrics[key] += (value * samples)
        total_samples += samples

    for key in aggregated_metrics.keys():
        aggregated_metrics[key] = round(aggregated_metrics[key] / total_samples, 6)

    return aggregated_metrics

if __name__ == "__main__":
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=1,
        min_evaluate_clients=1,
        min_available_clients=1,
        on_fit_config_fn=fit_config,
        evaluate_metrics_aggregation_fn=evaluate_metrics_aggregate,
        fit_metrics_aggregation_fn=fit_metrics_aggregate,
    )

    fl.server.start_server(
        server_address="127.0.0.1:8080",
        config=fl.server.ServerConfig(num_rounds=ROUNDS),
        strategy=strategy,
    )

    plt.bar(range(1, ROUNDS + 1), evaluate_accuracies, color='blue')
    plt.xlabel('Round')
    plt.ylabel('Evaluate Accuracy')
    plt.title('Federated Learning: Evaluate Accuracy over Rounds')
    plt.xticks(range(1, ROUNDS + 1))
    plt.grid(axis='y')
    plt.show()

