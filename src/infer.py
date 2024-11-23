import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import plotly.express as px

from models.components.imlenet import IMLENet, Config
from data.ptbxl_datamodule import PTBXLDataModule


import pyrootutils
pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)


def build_model(config, weights_path):
    model = IMLENet(
        input_channels=config.input_channels,
        signal_len=config.signal_len,
        beat_len=config.beat_len,
        start_filters=config.start_filters,
        kernel_size=config.kernel_size,
        num_blocks_list=config.num_blocks_list,
        lstm_units=config.lstm_units,
        classes=5,
    )
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.load_state_dict(torch.load(
        weights_path, map_location=device), strict=False)
    model.to(device)
    model.eval()
    return model


def build_scores(model, data, config):
    with torch.no_grad():
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        data = data.to(device)
        outputs, a_beat, a_rhythm, a_channel = model(data)
    print(torch.sigmoid(outputs))

    beat_att_weights = a_beat.squeeze(-1).cpu().numpy()
    rhythm_att_weights = a_rhythm.squeeze(-1).cpu().numpy()
    channel_att_weights = a_channel.squeeze(-1).cpu().numpy()

    # Beat scores
    lin = np.linspace(0, config.input_channels, num=config.beat_len)
    beat_att_weights = beat_att_weights.reshape(240, 13)
    beat_only = np.empty((240, config.beat_len))
    for i in range(beat_att_weights.shape[0]):
        beat_only[i] = np.interp(lin, np.arange(13), beat_att_weights[i])

    # Rhythm scores
    rhythm_att_weights = rhythm_att_weights.reshape(config.input_channels *
                            int(config.signal_len / config.beat_len))

    # Channel scores
    channel_att_weights = channel_att_weights.flatten()

    # Beat scores using channel
    beat_channel = np.copy(beat_only.reshape(
        config.input_channels, config.beat_len * int(config.signal_len / config.beat_len)))
    for i in range(config.input_channels):
        beat_channel[i] = beat_channel[i] * channel_att_weights[i]

    beat_normalized = (beat_channel.flatten() - beat_channel.flatten().min(keepdims=True)) / (
        beat_channel.flatten().max(keepdims=True) - beat_channel.flatten().min(keepdims=True))
    beat_normalized = beat_normalized.reshape(
        config.input_channels, config.signal_len)
    v_min = np.min(beat_channel.flatten())
    v_max = np.max(beat_channel.flatten())

    ch_info = ['I',
               'II',
               'III',
               'AVR',
               'AVL',
               'AVF',
               'V1',
               'V2',
               'V3',
               'V4',
               'V5',
               'V6']
    results_filepath = os.path.join(os.getcwd(), "results")
    os.makedirs(results_filepath, exist_ok=True)

    fig, axs = plt.subplots(config.input_channels, figsize=(35, 25))
    data = data.squeeze().cpu().numpy()

    for i, (ax, ch) in enumerate(zip(axs, ch_info)):
        im = ax.scatter(np.arange(len(
            data[i])), data[i], cmap='Spectral', c=beat_normalized[i], vmin=v_min, vmax=v_max)
        ax.plot(data[i], color=(0.2, 0.68, 1))
        ax.set_yticks([])
        ax.set_title(ch, fontsize=25)

    fig.tight_layout()
    cbar = fig.colorbar(im, ax=axs.ravel().tolist(), shrink=0.6)
    cbar.set_ticks([])
    plt.savefig(os.path.join(results_filepath, 'visualization.png'))

    fig = px.bar(channel_att_weights, title='Channel Importance Scores')
    fig.update_xaxes(tickvals=np.arange(
        config.input_channels), ticktext=ch_info)
    fig.write_html(os.path.join(results_filepath,
                   'channel_visualization.html'))



# @hydra.main(config_path="configs", config_name="eval.yaml")
# def main(cfg: DictConfig):
    # seed_everything(cfg.seed, workers=True)
def main():

    config = Config()
    weights_path = 'src\checkpoints\last.ckpt'
    model = build_model(config, weights_path)
    DL=PTBXLDataModule()
    DL.setup()
    data = DL.test_dataloader()
    data_iter = iter(data)
    batch = next(data_iter)
    inputs, targets = batch
    single_input = inputs[0]
    print(targets[0])
    single_input = single_input.unsqueeze(0) 
    build_scores(model, single_input, config)


if __name__ == "__main__":
    main()
