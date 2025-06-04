import mlx.nn as nn
import mlx.core as mx
import mlx.optimizers as optim
from functools import partial

from model import nge
from nge_utils import calculate_accuracy_metrics, SimpleLogger


class NGETrainer:    
    def __init__(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer
        self.state = [model.state, optimizer.state, mx.random.state]
        
        # Create gradient function for parallel training only
        self.parallel_loss_and_grad_fn = nn.value_and_grad(model, self._parallel_loss_fn)
        
        # Compile functions for parallel training only
        self.compiled_parallel_train_step = mx.compile(self._parallel_train_step_impl, inputs=self.state, outputs=self.state)
        self.compiled_parallel_eval_step = mx.compile(self._parallel_eval_step_impl, inputs=self.state, outputs=self.state)

    def _parallel_loss_fn(self, model, input_data, graph_targets, termination_target):
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

    def _parallel_train_step_impl(self, input_data, graph_targets, termination_target):
        (loss, losses, output, termination_prob), grads = self.parallel_loss_and_grad_fn(self.model, input_data, graph_targets, termination_target)
        
        self.optimizer.update(self.model, grads)
        return loss, losses, output, termination_prob

    def _parallel_eval_step_impl(self, input_data, graph_targets, termination_target):
        return self._parallel_loss_fn(self.model, input_data, graph_targets, termination_target)

    def train_step(self, input_data, graph_targets, termination_target, logger=None):
        result = self.compiled_parallel_train_step(input_data, graph_targets, termination_target)
        
        # Evaluate outside of compiled function
        mx.eval(result, self.model.parameters())
        
        # Handle logging outside of compiled function
        if logger:
            loss, losses, output, termination_prob = result
            state, distance, predesecor = output
            reachability_target, distance_target, predesecor_target = graph_targets
            
            # Calculate accuracy metrics
            metrics = calculate_accuracy_metrics(state, predesecor, reachability_target, predesecor_target, 
                                               termination_prob, termination_target, distance, distance_target)
            logger.log_step_metrics(metrics, phase='train', losses=losses)
        
        return result

    def eval_step(self, input_data, graph_targets, termination_target, logger=None):
        result = self.compiled_parallel_eval_step(input_data, graph_targets, termination_target)
        
        # Handle logging outside of compiled function
        if logger:
            loss, losses, output, termination_prob = result
            state, distance, predesecor = output
            reachability_target, distance_target, predesecor_target = graph_targets
            
            # Calculate accuracy metrics
            metrics = calculate_accuracy_metrics(state, predesecor, reachability_target, predesecor_target, 
                                               termination_prob, termination_target, distance, distance_target)
            logger.log_step_metrics(metrics, phase='val', losses=losses)
        
        return result

    def train_model(self, dataset, logger=None, phase="train"):
        """Training function for parallel algorithms (BFS)"""
        is_train = (phase == "train")
        total_loss = 0.0
        valid_graphs = 0
        
        for graph_idx, graph_data in enumerate(dataset):
            execution_history = graph_data['targets']['parallel']
            state_key = 'bfs_state'
            pred_key = 'bf_predecessor'
            term_key = 'bf_termination'
            distance_key = 'bf_distance'
            
            connection_matrix = graph_data['connection_matrix']
            residual_features = mx.zeros([len(execution_history[state_key][0])])
            num_steps = len(execution_history[state_key]) - 1

            if num_steps == 0:
                continue
            
            valid_graphs += 1
            graph_total_loss = 0.0
            
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
                
                # Training or evaluation step
                if is_train:
                    loss, losses, output, termination_prob = \
                        self.train_step(input_data, graph_targets, termination_target, logger)
                else:
                    loss, losses, output, termination_prob = \
                        self.eval_step(input_data, graph_targets, termination_target, logger)

                mx.eval(loss, losses, output, termination_prob)

                # Store debug info from the last step for epoch-level debugging
                if logger and logger.debug and i == num_steps - 1:  # Last step of this graph
                    state, distance, predesecor = output
                    reachability_target, distance_target, predesecor_target = graph_targets
                    from nge_utils import task
                    logger.store_debug_info(phase, state, predesecor, reachability_target, predesecor_target,
                                          termination_prob, termination_target, distance, distance_target, task.PARALLEL_ALGORITHM)

                # Update residual features
                state, distance, _ = output
                residual_features = mx.argmax(state, axis=1)
                graph_total_loss += float(loss)
            
            # Update progress bar
            avg_graph_loss = graph_total_loss / num_steps
            if logger:
                if is_train:
                    logger.update_progress(train_loss=avg_graph_loss)
                else:
                    logger.update_progress(val_loss=avg_graph_loss)
            
            total_loss += avg_graph_loss
        
        return total_loss / valid_graphs if valid_graphs > 0 else 0.0

    def train_harness(self, train_dataset, val_dataset, logger=None, 
                     num_epochs=10, early_stopping_patience=3):
        """Main training harness for parallel algorithms"""
        best_val_loss = float('inf')
        best_epoch = 0
        patience_counter = 0
        
        print("╭─ Training Progress")
        
        for epoch in range(num_epochs):
            # Start epoch progress bar
            if logger:
                total_graphs = len(train_dataset) + len(val_dataset)
                logger.start_epoch(epoch, num_epochs, total_graphs)
            
            # Training phase
            train_loss = self.train_model(train_dataset, logger, phase="train")
            
            # Validation phase
            val_loss = self.train_model(val_dataset, logger, phase="val")
            
            # Log epoch
            if logger:
                logger.log_epoch(epoch, train_loss, val_loss)
            
            # Check for improvement
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= early_stopping_patience:
                print(f"│  Early stopping at epoch {epoch + 1}")
                break
        
        # Log final results
        if logger:
            logger.log_final(best_val_loss, best_epoch)
        
        return {
            'best_val_loss': best_val_loss,
            'best_epoch': best_epoch,
            'train_loss': train_loss,
            'val_loss': val_loss
        }

    def evaluate_on_test_sets(self, test_datasets, logger=None):
        """Evaluate model on multiple test datasets"""
        results = {}
        
        for dataset_name, dataset in test_datasets.items():
            print(f"\nEvaluating on {dataset_name} ({len(dataset)} graphs)...")
            
            # Create a temporary logger to capture test metrics
            test_logger = SimpleLogger(debug=False) if logger is None else logger
            test_logger.start_epoch(0, 1, len(dataset))
            
            test_loss = self.train_model(dataset, test_logger, phase="test")
            
            # Get test accuracy metrics
            test_metrics = test_logger._average_metrics(test_logger.step_metrics.get('val', []))
            
            print(f"  parallel - Loss: {test_loss:.4f}", end="")
            if test_metrics:
                print(f", Acc - State: {test_metrics.get('state_acc', 0):.3f}, "
                      f"Pred: {test_metrics.get('pred_acc', 0):.3f}, "
                      f"Term: {test_metrics.get('term_acc', 0):.3f}, "
                      f"Dist: {test_metrics.get('dist_acc', 0):.3f}")
            else:
                print()
            
            results[dataset_name] = test_loss
        
        return results


def create_trainer(model_config, learning_rate=0.001):
    """Factory function to create a trainer with model and optimizer"""
    model = nge(**model_config)
    optimizer = optim.Adam(learning_rate=learning_rate)
    return NGETrainer(model, optimizer)