import warnings
import matplotlib.pyplot as plt
import numpy as np

from importlib.util import find_spec
from typing import Any, Callable, Dict, Optional, Tuple

from omegaconf import DictConfig

from src.utils import pylogger, rich_utils

log = pylogger.RankedLogger(__name__, rank_zero_only=True)


def extras(cfg: DictConfig) -> None:
    """Applies optional utilities before the task is started.

    Utilities:
        - Ignoring python warnings
        - Setting tags from command line
        - Rich config printing

    :param cfg: A DictConfig object containing the config tree.
    """
    # return if no `extras` config
    if not cfg.get("extras"):
        log.warning("Extras config not found! <cfg.extras=null>")
        return

    # disable python warnings
    if cfg.extras.get("ignore_warnings"):
        log.info("Disabling python warnings! <cfg.extras.ignore_warnings=True>")
        warnings.filterwarnings("ignore")

    # prompt user to input tags from command line if none are provided in the config
    if cfg.extras.get("enforce_tags"):
        log.info("Enforcing tags! <cfg.extras.enforce_tags=True>")
        rich_utils.enforce_tags(cfg, save_to_file=True)

    # pretty print config tree using Rich library
    if cfg.extras.get("print_config"):
        log.info("Printing config tree with Rich! <cfg.extras.print_config=True>")
        rich_utils.print_config_tree(cfg, resolve=True, save_to_file=True)


def task_wrapper(task_func: Callable) -> Callable:
    """Optional decorator that controls the failure behavior when executing the task function.

    This wrapper can be used to:
        - make sure loggers are closed even if the task function raises an exception (prevents multirun failure)
        - save the exception to a `.log` file
        - mark the run as failed with a dedicated file in the `logs/` folder (so we can find and rerun it later)
        - etc. (adjust depending on your needs)

    Example:
    ```
    @utils.task_wrapper
    def train(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        ...
        return metric_dict, object_dict
    ```

    :param task_func: The task function to be wrapped.

    :return: The wrapped task function.
    """

    def wrap(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        # execute the task
        try:
            metric_dict, object_dict = task_func(cfg=cfg)

        # things to do if exception occurs
        except Exception as ex:
            # save exception to `.log` file
            log.exception("")

            # some hyperparameter combinations might be invalid or cause out-of-memory errors
            # so when using hparam search plugins like Optuna, you might want to disable
            # raising the below exception to avoid multirun failure
            raise ex

        # things to always do after either success or exception
        finally:
            # display output dir path in terminal
            log.info(f"Output dir: {cfg.paths.output_dir}")

            # always close wandb run (even if exception occurs so multirun won't fail)
            if find_spec("wandb"):  # check if wandb is installed
                import wandb

                if wandb.run:
                    log.info("Closing wandb!")
                    wandb.finish()

        return metric_dict, object_dict

    return wrap


def get_metric_value(metric_dict: Dict[str, Any], metric_name: Optional[str]) -> Optional[float]:
    """Safely retrieves value of the metric logged in LightningModule.

    :param metric_dict: A dict containing metric values.
    :param metric_name: If provided, the name of the metric to retrieve.
    :return: If a metric name was provided, the value of the metric.
    """
    if not metric_name:
        log.info("Metric name is None! Skipping metric value retrieval...")
        return None

    if metric_name not in metric_dict:
        raise Exception(
            f"Metric value not found! <metric_name={metric_name}>\n"
            "Make sure metric name logged in LightningModule is correct!\n"
            "Make sure `optimized_metric` name in `hparams_search` config is correct!"
        )

    metric_value = metric_dict[metric_name].item()
    log.info(f"Retrieved metric value! <{metric_name}={metric_value}>")

    return metric_value

def draw_segmentation_timeline(ecg_signal, ecg_segment, length=5000, is_gt=False):
    """
    Draw segmentation timeline for a single ECG lead
    
    Parameters:
    -----------
    ecg_signal : numpy.ndarray
        Single lead ECG signal with shape (1, 5000)
    scg_segment : numpy.ndarray
        Segmentation data with shape (1, 4, 5000)
    length : int
        Length of signal to plot (default: 5000)
    """
    # Check input dimensions
    # if ecg_signal.shape[0] != 1 or ecg_segment.shape[1] != 4:
    #     print('Input should be for a single lead')
    #     return
        
    # Adjust length if signal is shorter than specified length
    if ecg_signal.shape[-1] < length:
        length = ecg_signal.shape[-1]
    
    if not is_gt:
        predicted_classes = np.argmax(ecg_segment, axis=0)

        # Convert to one-hot encoding
        ecg_segment = np.zeros((5000, 4))
        ecg_segment[np.arange(5000), predicted_classes] = 1
        ecg_segment = ecg_segment.T
    
    # Extract the segments for the single lead
      # shape becomes (4, 5000)
    
    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(36, 6))
    
    # Define colors for each segment
    colors = ['red', 'green', 'blue', 'yellow']
    
    # Create a single timeline
    timeline = np.zeros(length)
    for i in range(4):
        timeline[ecg_segment[i] == 1] = i + 1
    
    # Plot the timeline with colored segments
    for i in range(1, 5):
        ax.fill_between(range(length), -0.5, 1, 
                       where=timeline==i, 
                       facecolor=colors[i-1], 
                       alpha=0.3)
    
    # Plot the ECG signal
    plt.plot(np.arange(length), ecg_signal[0, :length], color='black')
    
    # Remove y-axis ticks and labels
    ax.set_yticks([])
    ax.set_yticklabels([])
    
    # Add labels and title
    plt.xlabel('Time (ms)')
    plt.ylabel('Amplitude')
    ax.set_title('ECG Segmentation Timeline')
    
    # Add a legend
    legend_elements = [plt.Rectangle((0,0),1,1, facecolor=colors[i], alpha=0.7) 
                      for i in range(4)]
    ax.legend(legend_elements, ['p', 'qrs', 't', 'None'],
             loc='upper center', 
             bbox_to_anchor=(0.5, -0.15), 
             ncol=4)
    
    # Add grid
    ax.grid(True, which='both', linestyle='--', color='gray', alpha=0.5)
    
    # Draw the canvas
    fig.canvas.draw()
    
    # Convert to numpy array
    data = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (4,))
    
    # Convert from RGBA to RGB
    rgb_data = data[:, :, :3]
    
    # Close the figure to free memory
    plt.close(fig)
    
    return rgb_data