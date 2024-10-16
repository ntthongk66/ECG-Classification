from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
    """A class used to build the feed-forward attention layer.

    Attributes
    ----------
    return_sequences: bool, optional
        If False, returns the calculated attention weighted sum of an ECG signal. (default: False)
    dim: int, optional
        The dimension of the attention layer. (default: 64)

    Methods
    -------
    forward(x)
        Calculates the attention weights.
    """

    def __init__(self, return_sequences: bool = False, dim: int = 64) -> None:
        super(Attention, self).__init__()
        self.return_sequences = return_sequences
        self.dim = dim
        
        # Weights for attention
        self.W = None
        self.b = None
        self.V = None

    def build(self, input_shape: torch.Size) -> None:
        """Builds the attention layer.

        alpha = softmax(V.T * tanh(W.T * x + b))

        Parameters
        ----------
        W: torch.Tensor
            The weights of the attention layer.
        b: torch.Tensor
            The bias of the attention layer.
        V: torch.Tensor
            The secondary weights of the attention layer.
        """
        self.W = nn.Parameter(torch.randn(input_shape[-1], self.dim)).to(torch.device('cuda'))
        self.b = nn.Parameter(torch.zeros(input_shape[1], self.dim)).to(torch.device('cuda'))
        self.V = nn.Parameter(torch.randn(self.dim, 1)).to(torch.device('cuda'))


    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Calculates the attention weights.

        Parameters
        ----------
        x: torch.Tensor
            The input tensor.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            The attention weighted sum of the input tensor and the attention weights.
        """
        if self.W is None:
            self.build(x.size())

        e = torch.tanh(torch.matmul(x, self.W) + self.b)
        e = torch.matmul(e, self.V)
        a = F.softmax(e, dim=1)
        output = x * a

        if self.return_sequences:
            return output, a

        return torch.sum(output, dim=1), a

class conv1d_tf(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(conv1d_tf, self).__init__()
        self.conv = nn.Conv1d(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding)
        
    def forward(self, input):
        out = self.conv(input)
        out = F.relu(out)
        return out

class ResidualBlock(nn.Module):
    """Residual block with optional downsampling."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        downsample: bool,
        kernel_size: int = 8,
    ):
        super(ResidualBlock, self).__init__()
        self.Downsample = downsample
        stride = 1 if not downsample else 2
        padding = self.calculate_padding(kernel_size, stride)
        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding
        )
        self.bn1 = nn.BatchNorm1d(out_channels, eps=0.001, momentum=0.01)
        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size, stride=1, padding=kernel_size // 2
        )
        

        self.bn2 = nn.BatchNorm1d(out_channels, eps=0.001, momentum=0.01)
        self.downsample = nn.Sequential()
        if downsample or in_channels != out_channels:
            # Adjust padding and stride for downsample convolution
            self.downsample = nn.Sequential(
                conv1d_tf(in_channels, out_channels, kernel_size=1, stride=stride, padding=0),
                nn.BatchNorm1d(out_channels, eps=0.001, momentum=0.01),
            )

    def calculate_padding(self, kernel_size: int, stride: int) -> int:
        # 'Same' padding formula for Conv1d
        padding = ((stride - 1) + kernel_size - stride) // 2
        return padding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)
        out = F.relu(out)
        out = self.bn1(out)
        out = self.conv2(out)
        x = self.downsample(x)
        out += x
        out = F.relu(out)
        out = self.bn2(out)
        return out


class IMLENet(nn.Module):
    """IMLE-Net model."""

    def __init__(
        self,
        input_channels,
        signal_len,
        beat_len,
        start_filters,
        kernel_size,
        num_blocks_list,
        lstm_units,
        classes,
        sub=False,
    ):
        super().__init__()
        self.sub = sub
        self.input_channels = input_channels
        self.signal_len = signal_len
        self.beat_len = beat_len
        self.start_filters = start_filters
        self.kernel_size = kernel_size
        self.num_blocks_list = num_blocks_list
        self.lstm_units = lstm_units
        self.classes = classes

        self.device = torch.device('cuda')
        # Beat Level
        # Calculate padding to achieve 'same' padding
        padding = ((self.kernel_size - 1) // 2)
        # self.beat_conv = nn.Conv1d(
        #     in_channels=1,
        #     out_channels=self.start_filters,
        #     kernel_size=self.kernel_size,
        #     padding=padding,
        # )
        
        self.beat_conv = nn.Conv1d(
            in_channels=1,
            out_channels=self.start_filters,
            kernel_size=self.kernel_size,
            stride=1,
            padding=padding
        )
        
        num_filters = self.start_filters

        # Residual Blocks
        blocks = []
        for i, num_blocks in enumerate(self.num_blocks_list):
            for j in range(num_blocks):
                downsample = j == 0 and i != 0
                out_channels = num_filters * (2 if downsample else 1)
                block = ResidualBlock(
                    in_channels=num_filters,
                    out_channels=out_channels,
                    downsample=downsample,
                    kernel_size=self.kernel_size,
                )
                blocks.append(block)
                num_filters = out_channels
        self.residual_blocks = nn.Sequential(*blocks)

        # Beat Attention
        # self.beat_attention = Attention(input_dim=num_filters)
        self.beat_attention = Attention()

        # Rhythm Level
        self.lstm = nn.LSTM(
            input_size=num_filters,
            hidden_size=self.lstm_units,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        self.rhythm_attention = Attention()
        
        # Channel Level

        self.channel_attention = Attention()
        
        # Output Layer
        self.fc = nn.Linear(self.lstm_units * 2, self.classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, input_channels, signal_len, 1)
        batch_size = x.size(0)
        x = x.squeeze(-1)  # (batch_size, input_channels, signal_len)
        x = x.view(-1, 1, self.beat_len)  # (batch_size * input_channels * num_beats, 1, beat_len)

        # Beat Level
        # print(x.shape) # (7680, 1, 50)
        x = self.beat_conv(x)
        # print(f"==== beat_conv1d {x.shape}") # (7680, 32, 49)
        x = F.relu(x)
        x = self.residual_blocks(x)
        x = x.transpose(
            1, 2
        )  # (batch_size * input_channels * num_beats, time_steps, features)

        x, _ = self.beat_attention(x)
        # print("attention", x.shape)
        
        # Rhythm Level
        num_beats = self.signal_len // self.beat_len
        
        # print("before reshape ", x.shape)
        x = x.view(batch_size * self.input_channels, num_beats, -1)
        
        # print('before: ', x.shape)
        x, _ = self.lstm(x)

        # print("after", x.shape)
        x, _ = self.rhythm_attention(x)

        # print("after rhymatt ", x.shape)
        # Channel Level
        x = x.view(batch_size, self.input_channels, -1)
        # print("before channel att: ", x.shape)
        x, _ = self.channel_attention(x)
        # print("after channel att: ", x.shape)

        # Output Layer
        x = self.fc(x)
        # outputs = torch.sigmoid(x)
        return x


"""Configs for building the IMLE-Net model.
"""


class Config:
    """A class used for IMLE-Net configs."""

    def __init__(self) -> None:
        """
        Parameters
        ----------
        signal_len: int
            The length of the input ECG signal (Time in secs * Sampling rate).
        input_channels: int
            The number of input channels of an ECG signal.
        beat_len: int
            The length of the segmented ECG beat (Time in secs * Sampling rate).
        kernel_size: int
            The kernel size of the 1D-convolution kernel.
        num_blocks_list: List[int]
            The number of residual blocks in the model.
        lstm_units: int
            The number of units in the LSTM layer.
        start_filters: int
            The number of filters at the start of the 1D-convolution layer.
        classes: int
        The number of classes in the output layer.

        """

        self.signal_len = 1000
        self.input_channels = 12
        self.beat_len = 50
        self.kernel_size = 8
        self.num_blocks_list = [2, 2, 2]
        self.lstm_units = 64
        self.start_filters = 32
        self.classes = 5


if __name__ == "__main__":
    config = Config()
    model = IMLENet(
        classes=config.classes,
        input_channels=config.input_channels,
        signal_len=config.signal_len,
        beat_len=config.beat_len,
        start_filters=config.start_filters,
        kernel_size=config.kernel_size,
        num_blocks_list=config.num_blocks_list,
        lstm_units=config.lstm_units,
    )
    input = torch.randn(32, 12, 1000, 1)
    print(input.size)
    # input = torch.from_numpy(input_np)
    output_torch = model(input)
    
    print(output_torch)
    print(torch.sigmoid(output_torch))
    
    pred = torch.argmax(output_torch, dim=1)
    # print(pred)
    # print(pred.dtype)
    # # print(output_torch)
    # output_tf = load_numpy(file_name="tf_output")
    # # print(output.shape)
    # # print(output)
    # compare("output", output_torch, output_tf)
