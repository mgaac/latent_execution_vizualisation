"""
Training script for Neural Graph Executor (NGE) model
Trains only on parallel algorithms (BFS) and saves the trained model
"""

import json
import pickle
import mlx.core as mx
from pathlib import Path

from model import aggregation_fn
from train import create_trainer
from nge_utils import SimpleLogger, print_model_info


# Model Configuration
MODEL_CONFIG = {
    'embedding_dim': 128,
    'dropout_prob': 0.5,
    'skip_connections': True,
    'aggregation_fn': aggregation_fn.MAX,
    'num_mp_layers': 3
}

# Training Hyperparameters
HYPERPARAMETERS = {
    'learning_rate': 0.0005,
    'num_epochs': 100,
    'early_stopping_patience': 10
}

# Paths
MODEL_SAVE_PATH = "saved_models"
MODEL_FILENAME = "nge_parallel_model"
DATASET_PATH = "dataset"


def load_dataset(dataset_path):
    """Load training and validation datasets"""
    print(f"╭─ Loading datasets from {dataset_path}")
    
    dataset_path_obj = Path(dataset_path)
    with open(dataset_path_obj / "train_graphs.pkl", "rb") as f:
        train_dataset = pickle.load(f)
    with open(dataset_path_obj / "val_graphs.pkl", "rb") as f:
        val_dataset = pickle.load(f)
    
    print(f"│  ✓ {len(train_dataset)} training graphs, {len(val_dataset)} validation graphs")

    return train_dataset, val_dataset


def save_model(trainer, model_config, hyperparameters, results, save_path):
    """Save the trained model using MLX export_function with shapeless support"""
    # Create save directory if it doesn't exist
    save_path_obj = Path(save_path)
    save_path_obj.mkdir(parents=True, exist_ok=True)
    
    # Evaluate the computation graph to ensure all parameters are materialized
    print("│  Evaluating computation graph...")
    mx.eval(trainer.model.parameters())
    
    # Create example inputs for export (matching your NGE model structure)
    # These should match the typical input shapes and dtypes your model expects
    dummy_input_features = mx.zeros((5, 2), dtype=mx.float32)  # [num_nodes, num_features]
    dummy_connection_matrix = mx.zeros((3, 5), dtype=mx.float32)  # [connections, nodes]
    
    # Export the model using MLX's export_function with shapeless=True
    export_file = str(save_path_obj / f"{MODEL_FILENAME}.mlxfn")

    #Generate export function
    model = trainer.model
    mx.eval(model.parameters())

    def model_export_fn(input_features, connection_matrix):
        # The model returns (output, termination_prob) where output is a tuple
        # For parallel algorithms: output = (bfs_state_predictions, bf_distance_predictions, predesecor_predictions)
        # We need to flatten this to a flat tuple of arrays for export_function
        (output, termination_prob) = model((input_features, connection_matrix))
        bfs_state_predictions, bf_distance_predictions, predesecor_predictions = output
        return (bfs_state_predictions, bf_distance_predictions, predesecor_predictions, termination_prob)
    
    print("│  Exporting model with shapeless support...")
    mx.export_function(
        export_file, 
        model_export_fn, 
        dummy_input_features, 
        dummy_connection_matrix,
        shapeless=True  # Enable shapeless export for variable input sizes
    )
    print(f"│  ✓ Model exported: {MODEL_FILENAME}.mlxfn")
    
    # Save configuration and metadata
    metadata = {
        'model_config': model_config,
        'hyperparameters': hyperparameters,
        'training_results': results,
        'model_class': 'nge',
        'input_structure': {
            'input_features_shape': list(dummy_input_features.shape),
            'connection_matrix_shape': list(dummy_connection_matrix.shape),
            'input_features_dtype': str(dummy_input_features.dtype),
            'connection_matrix_dtype': str(dummy_connection_matrix.dtype)
        },
        'output_structure': {
            'description': 'Returns flattened tuple: (bfs_state_predictions, bf_distance_predictions, predesecor_predictions, termination_prob)',
            'original_structure': 'Original model returns ((bfs_state, bf_distance, predesecor), termination)'
        }
    }
    
    config_file = str(save_path_obj / f"{MODEL_FILENAME}_config.json")
    with open(config_file, 'w') as f:
        # Convert non-serializable objects
        serializable_metadata = {}
        for key, value in metadata.items():
            if key == 'model_config' and 'aggregation_fn' in value:
                # Convert enum to its value
                config_copy = value.copy()
                config_copy['aggregation_fn'] = value['aggregation_fn'].value
                serializable_metadata[key] = config_copy
            elif key == 'training_results':
                # Convert any MLX arrays to Python types
                results_copy = {}
                for k, v in value.items():
                    if isinstance(v, mx.array):
                        results_copy[k] = float(v)
                    else:
                        results_copy[k] = v
                serializable_metadata[key] = results_copy
            else:
                serializable_metadata[key] = value
        
        json.dump(serializable_metadata, f, indent=2)
    
    print(f"│  ✓ Configuration saved: {MODEL_FILENAME}_config.json")
    print(f"│  Final validation loss: {results['best_val_loss']:.4f}")
    print(f"│  Best epoch: {results['best_epoch'] + 1}")


def load_model(save_path, model_filename=MODEL_FILENAME):
    """Load a saved model (for inference or continued training)"""
    save_path_obj = Path(save_path)
    export_file = str(save_path_obj / f"{model_filename}.mlxfn")
    config_file = str(save_path_obj / f"{model_filename}_config.json")
    weights_file = str(save_path_obj / f"{model_filename}.npz")
    
    config_file_path = Path(config_file)
    if not config_file_path.exists():
        raise FileNotFoundError(f"Model config file not found: {config_file}")
    
    # Load configuration
    with open(config_file, 'r') as f:
        metadata = json.load(f)
    
    # Recreate the model config with proper enum
    model_config = metadata['model_config'].copy()
    model_config['aggregation_fn'] = aggregation_fn(model_config['aggregation_fn'])
    
    imported_fn = mx.import_function(export_file)
    
    # Create a wrapper that reconstructs the original output structure
    def model_inference_fn(input_features, connection_matrix):
        """
        Wrapper that calls the imported function and reconstructs the original output structure.
        Returns: ((bfs_state_predictions, bf_distance_predictions, predesecor_predictions), termination_prob)
        """
        flat_output = imported_fn(input_features, connection_matrix)
        bfs_state_predictions, bf_distance_predictions, predesecor_predictions, termination_prob = flat_output
        output = (bfs_state_predictions, bf_distance_predictions, predesecor_predictions)
        return (output, termination_prob)
    
    print(f"│  ✓ Model loaded: {model_filename}.mlxfn")
    return model_inference_fn, metadata



def main():
    """Main training function"""
    print("╭─" + "─" * 30 + "─╮")
    print("│ Neural Graph Executor Training │")
    print("╰─" + "─" * 30 + "─╯")
    print()
    
    # Load datasets
    train_dataset, val_dataset = load_dataset(DATASET_PATH)
    
    # Create trainer
    print("\n╭─ Model Setup")
    trainer = create_trainer(MODEL_CONFIG, learning_rate=HYPERPARAMETERS['learning_rate'])
    
    # Print model information
    print_model_info(trainer.model)
    
    # Create logger
    logger = SimpleLogger(debug=False)
    
    # Start training
    print("\n╭─ Training Configuration")
    print(f"│  Epochs: {HYPERPARAMETERS['num_epochs']}")
    print(f"│  Early stopping patience: {HYPERPARAMETERS['early_stopping_patience']}")
    print(f"│  Learning rate: {HYPERPARAMETERS['learning_rate']}")
    print()
    
    results = trainer.train_harness(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        logger=logger,
        num_epochs=HYPERPARAMETERS['num_epochs'],
        early_stopping_patience=HYPERPARAMETERS['early_stopping_patience']
    )
    
    # Save the trained model
    print("\n╭─ Model Export")
    save_model(trainer, MODEL_CONFIG, HYPERPARAMETERS, results, MODEL_SAVE_PATH)

if __name__ == "__main__":
    # Set random seed for reproducibility
    mx.random.seed(42)
    
    # Run training
    main() 