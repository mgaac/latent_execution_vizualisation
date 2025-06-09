import mlx.core as mx
import mlx.nn as nn
import numpy as np
from enum import Enum

# Task enumeration (needed for logging)
class task(Enum):
    PARALLEL_ALGORITHM=0
    SEQUENTIAL_ALGORITHM=1

def calculate_accuracy_metrics(state, predesecor, reachability_target, predesecor_target, 
                             termination_prob, termination_target, distance=None, distance_target=None):
    """Calculate accuracy metrics for all task components"""
    metrics = {}
    
    # State accuracy (binary classification)
    state_pred = mx.argmax(state, axis=1)
    state_targ = mx.argmax(reachability_target, axis=1)
    state_acc = float(mx.mean((state_pred == state_targ).astype(mx.float32)))
    metrics['state_acc'] = state_acc
    
    # Predecessor accuracy (multi-class classification)
    pred_pred = mx.argmax(predesecor, axis=1)
    pred_acc = float(mx.mean((pred_pred == predesecor_target).astype(mx.float32)))
    metrics['pred_acc'] = pred_acc
    
    # Termination accuracy (binary classification)
    term_pred = mx.argmax(termination_prob, axis=0 if termination_prob.ndim == 1 else 1)
    term_targ = mx.argmax(termination_target, axis=0 if termination_target.ndim == 1 else 1)
    term_acc = float((term_pred == term_targ).astype(mx.float32))
    metrics['term_acc'] = term_acc
    
    # Distance accuracy (for parallel algorithms only)
    if distance is not None and distance_target is not None:
        # For distance, we calculate MAE and also a threshold-based accuracy
        distance_mae = float(mx.mean(mx.abs(distance.flatten() - distance_target.flatten())))
        # Consider distance "accurate" if within 10% of target (or 0.1 absolute for small values)
        distance_thresh = mx.maximum(mx.abs(distance_target.flatten()) * 0.15, 0.15)
        distance_within_thresh = mx.abs(distance.flatten() - distance_target.flatten()) <= distance_thresh
        distance_acc = float(mx.mean(distance_within_thresh.astype(mx.float32)))
        metrics['dist_mae'] = distance_mae
        metrics['dist_acc'] = distance_acc
    
    return metrics

class SimpleLogger:
    """Minimal logger with optional debug mode and accuracy tracking"""
    def __init__(self, debug=False):
        self.debug = debug
        # Track accuracy metrics for each epoch
        self.train_metrics = []
        self.val_metrics = []
        self.step_metrics = {'train': [], 'val': []}
        # Track individual loss components
        self.step_losses = {'train': [], 'val': []}
        # Store latest debug info for epoch-level printing
        self.latest_debug_info = {'train': None, 'val': None}
    
    def start_epoch(self, epoch, num_epochs, total_graphs):
        """Start epoch (no progress bar)"""
        # Reset step metrics and losses for new epoch
        self.step_metrics = {'train': [], 'val': []}
        self.step_losses = {'train': [], 'val': []}
        # Reset debug info for new epoch
        self.latest_debug_info = {'train': None, 'val': None}
    
    def update_progress(self, train_loss=None, val_loss=None):
        """Update progress (no progress bar)"""
        pass
    
    def log_step_metrics(self, metrics, phase='train', losses=None):
        """Log metrics and losses for a single step"""
        self.step_metrics[phase].append(metrics)
        if losses is not None:
            # losses should be a tuple: (state_loss, distance_loss, pred_loss, termination_loss)
            loss_dict = {
                'state_loss': float(losses[0]),
                'distance_loss': float(losses[1]),
                'pred_loss': float(losses[2]),
                'termination_loss': float(losses[3])
            }
            self.step_losses[phase].append(loss_dict)
    
    def store_debug_info(self, phase, state, predesecor, reachability_target, predesecor_target, 
                        termination_prob, termination_target, distance=None, distance_target=None, task_type=None):
        """Store debug information to be printed at epoch level"""
        if self.debug:
            self.latest_debug_info[phase] = {
                'state': state,
                'predesecor': predesecor,
                'reachability_target': reachability_target,
                'predesecor_target': predesecor_target,
                'termination_prob': termination_prob,
                'termination_target': termination_target,
                'distance': distance,
                'distance_target': distance_target,
                'task_type': task_type
            }
    
    def _average_metrics(self, metrics_list):
        """Average a list of metric dictionaries"""
        if not metrics_list:
            return {}
        
        # Get all keys from first metrics dict
        keys = metrics_list[0].keys()
        averaged = {}
        
        for key in keys:
            values = [m[key] for m in metrics_list if key in m]
            if values:
                averaged[key] = sum(values) / len(values)
        
        return averaged
    
    def _average_losses(self, losses_list):
        """Average a list of loss dictionaries"""
        if not losses_list:
            return {}
        
        # Get all keys from first loss dict
        keys = losses_list[0].keys()
        averaged = {}
        
        for key in keys:
            values = [l[key] for l in losses_list if key in l]
            if values:
                averaged[key] = sum(values) / len(values)
        
        return averaged
    
    def log_epoch(self, epoch, train_loss, val_loss):
        """Log epoch results with accuracy metrics and component losses"""
        # Calculate average metrics for the epoch
        train_metrics = self._average_metrics(self.step_metrics['train'])
        val_metrics = self._average_metrics(self.step_metrics['val'])
        
        # Calculate average losses for the epoch
        train_losses = self._average_losses(self.step_losses['train'])
        val_losses = self._average_losses(self.step_losses['val'])
        
        # Store metrics
        self.train_metrics.append(train_metrics)
        self.val_metrics.append(val_metrics)
        
        # Print epoch summary with elegant formatting and aligned dividers
        print(f"│  Epoch {epoch + 1}")
        print(f"│    Losses      Train: {train_loss:.3f}  │  Val: {val_loss:.3f}   │")
        
        
        # Print component-wise losses if available with aligned dividers
        if train_losses:
            state_loss = train_losses.get('state_loss', 0)
            pred_loss = train_losses.get('pred_loss', 0)
            term_loss = train_losses.get('termination_loss', 0)
            dist_loss = train_losses.get('distance_loss', 0)
            print(f"│    Train Loss  State: {state_loss:.3f}  │  Pred: {pred_loss:.3f}  │  Term: {term_loss:.3f}  │  Dist: {dist_loss:.3f}")
        
        if val_losses:
            state_loss = val_losses.get('state_loss', 0)
            pred_loss = val_losses.get('pred_loss', 0)
            term_loss = val_losses.get('termination_loss', 0)
            dist_loss = val_losses.get('distance_loss', 0)
            print(f"│    Val Loss    State: {state_loss:.3f}  │  Pred: {pred_loss:.3f}  │  Term: {term_loss:.3f}  │  Dist: {dist_loss:.3f}")
        
        # Print accuracy metrics if available with aligned dividers
        if train_metrics:
            state_acc = train_metrics.get('state_acc', 0)
            pred_acc = train_metrics.get('pred_acc', 0)
            term_acc = train_metrics.get('term_acc', 0)
            print(f"│    Train Acc   State: {state_acc:.3f}  │  Pred: {pred_acc:.3f}  │  Term: {term_acc:.3f}", end="")
            if 'dist_acc' in train_metrics:
                dist_acc = train_metrics.get('dist_acc', 0)
                dist_mae = train_metrics.get('dist_mae', 0)
                print(f"  │  Dist: {dist_acc:.3f} (MAE: {dist_mae:.3f})")
            else:
                print()
        
        if val_metrics:
            state_acc = val_metrics.get('state_acc', 0)
            pred_acc = val_metrics.get('pred_acc', 0)
            term_acc = val_metrics.get('term_acc', 0)
            print(f"│    Val Acc     State: {state_acc:.3f}  │  Pred: {pred_acc:.3f}  │  Term: {term_acc:.3f}", end="")
            if 'dist_acc' in val_metrics:
                dist_acc = val_metrics.get('dist_acc', 0)
                dist_mae = val_metrics.get('dist_mae', 0)
                print(f"  │  Dist: {dist_acc:.3f} (MAE: {dist_mae:.3f})")
            else:
                print()
        
        # Print debug info if available and debug mode is enabled
        if self.debug:
            for phase in ['train', 'val']:
                debug_info = self.latest_debug_info.get(phase)
                if debug_info:
                    self._print_debug_info(phase, debug_info)
        
        print("│")
    
    def _print_debug_info(self, phase, debug_info):
        """Print debug information for a given phase"""
        state = debug_info['state']
        predesecor = debug_info['predesecor']
        reachability_target = debug_info['reachability_target']
        predesecor_target = debug_info['predesecor_target']
        termination_prob = debug_info['termination_prob']
        termination_target = debug_info['termination_target']
        distance = debug_info['distance']
        distance_target = debug_info['distance_target']
        task_type = debug_info['task_type']
        
        task_name = "SEQ" if task_type == task.SEQUENTIAL_ALGORITHM else "PAR"
        phase_name = phase.upper()
        print(f"│    DEBUG ({phase_name}-{task_name}):")
        print(f"│      Pred: {np.array(mx.argmax(predesecor, axis=1))}")
        print(f"│      Targ: {np.array(predesecor_target)}")
        print(f"│      State: {np.array(mx.argmax(state, axis=1))}")
        print(f"│      S_Targ: {np.array(mx.argmax(reachability_target, axis=1))}")
        
        if distance is not None and distance_target is not None:
            print(f"│      Dist: {np.array(distance.flatten())}")
            print(f"│      D_Targ: {np.array(distance_target.flatten())}")
        
        print(f"│      Term: {float(mx.softmax(termination_prob, axis=0)[1]):.3f}")
    
    def log_debug_info(self, state, predesecor, reachability_target, predesecor_target, 
                      termination_prob, termination_target, distance=None, distance_target=None, task_type=None):
        """Legacy method - now just stores debug info for epoch-level printing"""
        # This method is kept for backward compatibility but now just stores the info
        # The actual printing happens in log_epoch
        pass
    
    def log_final(self, best_val_loss, best_epoch):
        """Log final training results"""
        print(f"│  Best validation loss: {best_val_loss:.4f} at epoch {best_epoch + 1}")
        
        # Print best validation accuracies if available
        if best_epoch < len(self.val_metrics) and self.val_metrics[best_epoch]:
            best_metrics = self.val_metrics[best_epoch]
            print(f"│  Best validation accuracies:")
            state_acc = best_metrics.get('state_acc', 0)
            pred_acc = best_metrics.get('pred_acc', 0)
            term_acc = best_metrics.get('term_acc', 0)
            print(f"│    State: {state_acc:.3f}  │  Pred: {pred_acc:.3f}  │  Term: {term_acc:.3f}", end="")
            if 'dist_acc' in best_metrics:
                dist_acc = best_metrics.get('dist_acc', 0)
                print(f"  │  Dist: {dist_acc:.3f}")
            else:
                print()