import numpy as np
from src.utils import draw_segmentation_timeline

def find_boundaries(signal) -> tuple:
    """
    Find onsets and offsets by detecting transitions in the signal.
    
    Parameters:
    -----------
    signal : numpy.ndarray
        Binary signal where 1 indicates the presence of a wave
        
    Returns:
    --------
    tuple
        Arrays of onset and offset positions
    """
    transitions = np.diff(signal)
    onsets = np.where(transitions == 1)[0] + 1
    offsets = np.where(transitions == -1)[0] + 1
    return onsets, offsets

def calculate_f1_score(sensitivity, ppv):
    """
    Calculate F1 score from sensitivity and PPV
    """
    if sensitivity + ppv == 0:
        return 0
    return 2 * (sensitivity * ppv) / (sensitivity + ppv)

def calculate_wave_metrics(y_true, y_pred, tolerance=0) -> dict:
    """
    Calculate sensitivity, PPV, and F1 score for ECG wave boundaries.
    
    Parameters:
    -----------
    y_true : numpy.ndarray
        Ground truth labels with shape (batch_size, 3, sequence_length)
        Each channel represents one wave component (P, QRS, T)
    y_pred : numpy.ndarray
        Predicted labels with shape (batch_size, 3, sequence_length)
    tolerance : int, optional
        Number of samples tolerance for matching predictions with ground truth
        
    Returns:
    --------
    dict
        Dictionary containing sensitivity, PPV, and F1 score for each wave component's onset and offset
    """
    
    # pre process for y_pred
    batch_size, num_classes, seq_length = y_pred.shape
    ecg_batch = np.random.rand(batch_size, num_classes, seq_length)

    # Find predicted classes for each element in the batch
    predicted_classes = np.argmax(ecg_batch, axis=1)  # Shape: (batch_size, seq_length)

    # Convert to one-hot encoding
    ecg_batch_one_hot = np.zeros((batch_size, num_classes, seq_length))
    ecg_batch_one_hot[np.arange(batch_size)[:, None], predicted_classes, np.arange(seq_length)] = 1

    y_pred = ecg_batch_one_hot

    
    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shapes must match. Got y_true: {y_true.shape}, y_pred: {y_pred.shape}")
    
    batch_size, n_channels, seq_length = y_true.shape
    wave_components = ['P', 'QRS', 'T']
    metrics = {}
    
    # Initialize total counters for average calculation
    total_tp = 0
    total_fp = 0
    total_fn = 0
    
    for channel_idx, wave in enumerate(wave_components):
        metrics[f'{wave}_onset'] = {'tp': 0, 'fp': 0, 'fn': 0}
        metrics[f'{wave}_offset'] = {'tp': 0, 'fp': 0, 'fn': 0}
        
        for batch in range(batch_size):
            true_onsets, true_offsets = find_boundaries(y_true[batch, channel_idx])
            pred_onsets, pred_offsets = find_boundaries(y_pred[batch, channel_idx])
            
            # Match onsets
            matched_true_onsets = set()
            matched_pred_onsets = set()
            
            for pred_pos in pred_onsets:
                found_match = False
                for true_pos in true_onsets:
                    if true_pos not in matched_true_onsets:
                        if abs(pred_pos - true_pos) <= tolerance:
                            metrics[f'{wave}_onset']['tp'] += 1
                            matched_true_onsets.add(true_pos)
                            matched_pred_onsets.add(pred_pos)
                            found_match = True
                            break
                if not found_match:
                    metrics[f'{wave}_onset']['fp'] += 1
            
            metrics[f'{wave}_onset']['fn'] += len(true_onsets) - len(matched_true_onsets)
            
            # Match offsets
            matched_true_offsets = set()
            matched_pred_offsets = set()
            
            for pred_pos in pred_offsets:
                found_match = False
                for true_pos in true_offsets:
                    if true_pos not in matched_true_offsets:
                        if abs(pred_pos - true_pos) <= tolerance:
                            metrics[f'{wave}_offset']['tp'] += 1
                            matched_true_offsets.add(true_pos)
                            matched_pred_offsets.add(pred_pos)
                            found_match = True
                            break
                if not found_match:
                    metrics[f'{wave}_offset']['fp'] += 1
            
            metrics[f'{wave}_offset']['fn'] += len(true_offsets) - len(matched_true_offsets)
    
    # Calculate metrics for each boundary type and accumulate totals
    results = {}
    
    for point, values in metrics.items():
        sensitivity = values['tp'] / (values['tp'] + values['fn']) if (values['tp'] + values['fn']) > 0 else 0
        ppv = values['tp'] / (values['tp'] + values['fp']) if (values['tp'] + values['fp']) > 0 else 0
        f1_score = calculate_f1_score(sensitivity, ppv)
        
        results[point] = {
            'sensitivity': sensitivity,
            'ppv': ppv,
            'f1_score': f1_score,
            'true_positives': values['tp'],
            'false_positives': values['fp'],
            'false_negatives': values['fn']
        }
        
        # Accumulate totals
        total_tp += values['tp']
        total_fp += values['fp']
        total_fn += values['fn']
    
    # Calculate average metrics using total counts
    avg_sensitivity = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    avg_ppv = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    avg_f1_score = calculate_f1_score(avg_sensitivity, avg_ppv)
    
    # Add average metrics to results
    results['average'] = {
        'sensitivity': avg_sensitivity,
        'ppv': avg_ppv,
        'f1_score': avg_f1_score,
        'true_positives': total_tp,
        'false_positives': total_fp,
        'false_negatives': total_fn
    }
    
    return results

def print_wave_metrics(metrics):
    """
    Print wave component metrics in a formatted way
    """
    print("\nECG Wave Boundary Detection Metrics:")
    print("-" * 50)
    
    wave_order = ['P_onset', 'P_offset', 'QRS_onset', 'QRS_offset', 'T_onset', 'T_offset', 'average']
    
    for component in wave_order:
        values = metrics[component]
        print(f"\n{component.replace('_', ' ').title()}:")
        print(f"  Sensitivity: {values['sensitivity']:.4f}")
        print(f"  PPV:        {values['ppv']:.4f}")
        print(f"  F1 Score:   {values['f1_score']:.4f}")
        if component != 'average':
            print(f"  True Positives:  {values['true_positives']}")
            print(f"  False Positives: {values['false_positives']}")
            print(f"  False Negatives: {values['false_negatives']}")

def calculate_metrics(tp, fp, fn) -> dict:
    """
    Calculate Sensitivity, PPV, and F1 score.

    Parameters:
        tp (int): True Positives
        fp (int): False Positives
        fn (int): False Negatives

    Returns:
        dict: A dictionary containing sensitivity, ppv, and f1_score.
    """
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
    f1_score = 2 * (sensitivity * ppv) / (sensitivity + ppv) if (sensitivity + ppv) > 0 else 0

    return {
        "sensitivity": sensitivity,
        "ppv": ppv,
        "f1_score": f1_score
    }

if __name__ == '__main__':
    pred = np.load('/work/hpc/ntt/ECG-Classification/output/pred/0.npy')[:, :, 500:3500] # (16, 4, 3000)
    tg = np.load('/work/hpc/ntt/ECG-Classification/output/gt/0.npy')[:, :, 500:3500]
    # signal = np.load()
     
    metrics = calculate_wave_metrics(tg, pred, tolerance=75)
    print_wave_metrics(metrics)