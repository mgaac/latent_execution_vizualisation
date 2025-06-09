import mlx.core as mx
import mlx.nn as nn

import numpy as np

from nge_utils import calculate_accuracy_metrics
from train_script import load_model, load_dataset

# Load the trained model using the proper loading function
model, metadata = load_model("saved_models")
train_dataset, val_dataset, test_datasets = load_dataset("dataset")


def parallel_loss_fn(model, input_data, graph_targets, termination_target):
    """Loss function for parallel algorithms (BFS)"""
    output, termination_prob = model(input_data)
    state, distance, predesecor = output
    reachability_target, distance_target, predesecor_target = graph_targets

    state_loss = nn.losses.binary_cross_entropy(state, reachability_target, reduction='mean')
    distance_loss = nn.losses.mse_loss(distance.squeeze(), distance_target, reduction='mean')
    pred_loss = nn.losses.cross_entropy(predesecor, predesecor_target, reduction='mean')
    termination_loss = nn.losses.binary_cross_entropy(termination_prob, termination_target, reduction='mean')
    total_loss = state_loss + distance_loss + pred_loss + termination_loss

    return total_loss, (state_loss, distance_loss, pred_loss, termination_loss), output, termination_prob


def eval_step(model, input_data, graph_targets, termination_target):
    result = parallel_loss_fn(model, input_data, graph_targets, termination_target)
    
    loss, losses, output, termination_prob = result
    state, distance, predesecor = output
    reachability_target, distance_target, predesecor_target = graph_targets
    
    # Calculate accuracy metrics
    metrics = calculate_accuracy_metrics(state, predesecor, reachability_target, predesecor_target, 
                                        termination_prob, termination_target, distance, distance_target)
    
    return result

def inspect_execution_trace(graph_data):
    """Display the execution trace targets from the dataset with elegant formatting"""
    execution_history = graph_data['targets']['parallel']
    state_key = 'bfs_state'
    pred_key = 'bf_predecessor'
    term_key = 'bf_termination'
    distance_key = 'bf_distance'
    
    num_steps = len(execution_history[state_key]) - 1
    
    for i in range(num_steps + 1):  # Include initial state
        state_target = execution_history[state_key][i]
        pred_target = execution_history[pred_key][i]
        termination_target = execution_history[term_key][i]
        distance_target = execution_history[distance_key][i]
        
        print(f"\nStep {i:2d}")
        print("─" * 50)
        
        # State
        state_true = np.argmin(state_target, axis=1)
        print(f"State       True: {state_true}")
        
        # Distance
        dist_true = np.round(np.array(distance_target), 3)
        # print(f"Distance    True: {dist_true}")
        
        # Predecessor
        pred_true = np.array(pred_target)
        print(f"Predecessor True: {pred_true}")
        
        # Termination
        term_true = termination_target
        print(f"Termination True: {term_true}")


def execute_graph_algorithm(model, graph_data):
    execution_history = graph_data['targets']['parallel']
    state_key = 'bfs_state'
    pred_key = 'bf_predecessor'
    term_key = 'bf_termination'
    distance_key = 'bf_distance'
    
    connection_matrix = graph_data['connection_matrix']
    residual_features = mx.zeros([len(execution_history[state_key][0])])
    num_steps = len(execution_history[state_key]) - 1

    for i in range(num_steps):
        # Prepare data
        state_target = execution_history[state_key][i + 1]
        pred_target = execution_history[pred_key][i + 1]
        termination_target = execution_history[term_key][i + 1]
        distance_target = execution_history[distance_key][i + 1]
        
        current_features = mx.argmax(execution_history[state_key][i], axis=1)
        input_features = mx.stack([current_features, residual_features], axis=1)
        input_data = (input_features, connection_matrix)
        
        graph_targets = (state_target, distance_target, pred_target)
        
        loss, losses, output, termination_prob = eval_step(model, input_data, graph_targets, termination_target)


        state, distance, predesecor = output
        reachability_target, distance_target, predesecor_target = graph_targets

        # Print execution step with elegant formatting
        print(f"\nStep {i:2d}")
        print("─" * 50)
        
        # State comparison
        state_pred = np.argmin(state, axis=1) 
        state_true = np.argmin(reachability_target, axis=1)
        print(f"State       Pred: {state_pred}")
        print(f"            True: {state_true}")
        
        # Distance comparison
        # dist_pred = np.round(np.array(distance.squeeze()), 3)
        # dist_true = np.round(np.array(distance_target.squeeze()), 3)
        # print(f"Distance    Pred: {dist_pred}")
        # print(f"            True: {dist_true}")
        
        # Predecessor comparison
        pred_pred = np.argmax(predesecor, axis=1)
        pred_true = np.array(predesecor_target)
        print(f"Predecessor Pred: {pred_pred}")
        print(f"            True: {pred_true}")
        
        # Termination comparison
        term_prob = float(mx.softmax(termination_prob, axis=0)[0])
        term_true = int(mx.argmax(termination_target, axis=0))
        print(f"Termination Pred: {term_prob:.4f}")
        print(f"            True: {term_true}")
        
        # Loss summary
        print(f"Loss        Total: {float(loss):.6f}")
        print(f"            (S:{float(losses[0]):.4f} D:{float(losses[1]):.4f} P:{float(losses[2]):.4f} T:{float(losses[3]):.4f})")


        # Update residual features
        state, distance, _ = output
        residual_features = mx.argmax(state, axis=1)




# Inspect the execution trace targets
inspect_execution_trace(test_datasets["test_20"][0])

# print("\n" + "═" * 60)
# print("MODEL EXECUTION")
# print("═" * 60)

# execute_graph_algorithm(model, test_datasets["test_20"][0])        


