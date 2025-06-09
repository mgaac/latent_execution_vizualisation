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
from nge_utils import SimpleLogger


# Model Configuration
MODEL_CONFIG = {
    'embedding_dim': 32,
    'dropout_prob': 0.1,
    'skip_connections': True,
    'aggregation_fn': aggregation_fn.MAX,
    'num_mp_layers': 2
}

# Training Hyperparameters
HYPERPARAMETERS = {
    'learning_rate': 0.0005,
    'num_epochs': 30,
    'early_stopping_patience': 10
}

# Paths
MODEL_SAVE_PATH = "saved_models"
MODEL_FILENAME = "nge_parallel_model"
DATASET_PATH = "dataset"


def load_dataset(dataset_path):
    """Load training, validation, and test datasets"""
    # print(f"\n╭─ Loading datasets from {dataset_path}")
    
    dataset_path_obj = Path(dataset_path)
    with open(dataset_path_obj / "train_graphs.pkl", "rb") as f:
        train_dataset = pickle.load(f)
    
    # Check for val_graphs.pkl in dataset directory first, then root
    val_file_path = dataset_path_obj / "val_graphs.pkl"
    if not val_file_path.exists():
        val_file_path = Path("val_graphs.pkl")
    
    with open(val_file_path, "rb") as f:
        val_dataset = pickle.load(f)
    
    # Load test datasets
    test_datasets = {}
    test_files = ["test_graphs_20.pkl", "test_graphs_50.pkl", "test_graphs_100.pkl"]
    
    for test_file in test_files:
        test_path = dataset_path_obj / test_file
        if test_path.exists():
            with open(test_path, "rb") as f:
                test_dataset = pickle.load(f)
                # Extract the size from filename for cleaner naming
                size = test_file.replace("test_graphs_", "").replace(".pkl", "")
                test_datasets[f"test_{size}"] = test_dataset
                # print(f"│  ✓ {len(test_dataset)} test graphs ({size} graphs)")
    
    # print(f"│  ✓ {len(train_dataset)} training graphs, {len(val_dataset)} validation graphs")
    
    return train_dataset, val_dataset, test_datasets


def save_model(trainer, model_config, hyperparameters, results, save_path):
    """Save the trained model weights using MLX save_weights"""
    # Create save directory if it doesn't exist
    save_path_obj = Path(save_path)
    save_path_obj.mkdir(parents=True, exist_ok=True)
    
    # Evaluate the computation graph to ensure all parameters are materialized
    print("│  Evaluating computation graph...")
    mx.eval(trainer.model.parameters())
    
    # Save model weights
    weights_file = str(save_path_obj / f"{MODEL_FILENAME}.npz")
    print("│  Saving model weights...")
    trainer.model.save_weights(weights_file)
    print(f"│  ✓ Model weights saved: {MODEL_FILENAME}.npz")
    
    # Save configuration and metadata
    metadata = {
        'model_config': model_config,
        'hyperparameters': hyperparameters,
        'training_results': results,
        'model_class': 'nge'
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
    config_file = str(save_path_obj / f"{model_filename}_config.json")
    weights_file = str(save_path_obj / f"{model_filename}.npz")
    
    config_file_path = Path(config_file)
    if not config_file_path.exists():
        raise FileNotFoundError(f"Model config file not found: {config_file}")
    
    weights_file_path = Path(weights_file)
    if not weights_file_path.exists():
        raise FileNotFoundError(f"Model weights file not found: {weights_file}")
    
    # Load configuration
    with open(config_file, 'r') as f:
        metadata = json.load(f)
    
    # Recreate the model config with proper enum
    model_config = metadata['model_config'].copy()
    model_config['aggregation_fn'] = aggregation_fn(model_config['aggregation_fn'])
    
    # Create model instance
    from model import nge
    model = nge(**model_config)
    
    # Load weights
    # print("│  Loading model weights...")
    model.load_weights(weights_file)
    # print(f"│  ✓ Model weights loaded: {model_filename}.npz")
    
    return model, metadata



def main():
    """Main training function"""
    print("╭─" + "─" * 30 + "─╮")
    print("│ Neural Graph Executor Training │")
    print("╰─" + "─" * 30 + "─╯")
    print()
    

    train_dataset, val_dataset, test_datasets = load_dataset(DATASET_PATH)
    
    trainer = create_trainer(MODEL_CONFIG, learning_rate=HYPERPARAMETERS['learning_rate'])
    
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
    
    # Evaluate on test datasets
    if test_datasets:
        print("\n╭─ Test Dataset Evaluation")
        test_results = trainer.evaluate_on_test_sets(test_datasets, logger)
        
        # Add test results to the results dictionary and save updated config
        results['test_results'] = test_results
        
        # Re-save the model with test results included
        save_model(trainer, MODEL_CONFIG, HYPERPARAMETERS, results, MODEL_SAVE_PATH)
        
        print("│  Test evaluation complete!")
        print("╰─" + "─" * 30 + "─╯")

if __name__ == "__main__":
    # Set random seed for reproducibility
    mx.random.seed(42)
    
    # Run training
    main() 