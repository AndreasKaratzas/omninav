
import sys
sys.path.append("../")

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.noisy import NoisyLinear


def init_(module, weight_init=nn.init.orthogonal_, bias_init=nn.init.constant_, gain=1):
    """Initializes module with orthogonal weight initialization and constant bias initialization.

    Parameters
    ----------
    module : torch.nn.Module
        Module to be initialized.
    weight_init : function (default: nn.init.orthogonal_)
        Weight initialization function.
    bias_init : function (default: nn.init.constant_)
        Bias initialization function.
    gain : float (default: 1)
        Gain value for weight initialization.

    Returns
    -------
    torch.nn.Module
        Initialized module.
    """
    weight_init(module.weight.data, gain=gain)
    if module.bias is not None:
        bias_init(module.bias.data)
    return module


def init(module, activation='relu'):
    """Initializes module with orthogonal weight initialization and constant bias initialization.

    Parameters
    ----------
    module : torch.nn.Module
        Module to be initialized.
    activation : str, optional
        Activation function used in the module, by default 'relu'

    Returns
    -------
    torch.nn.Module
        Initialized module.
    """
    activation_gain = {
        'relu': nn.init.calculate_gain('relu'),
        'gelu': 1.0
    }

    gain = activation_gain[activation]
    return init_(module,
                 nn.init.orthogonal_,
                 lambda x: nn.init.constant_(x, 0),
                 gain)


class RainbowDQN(nn.Module):
    """Q model implementation of Rainbow agent.

    Attributes
    ----------
    in_dim : int
        Number of elements expected in the input vector.
    out_dim : int
        Number of elements at the output layer corresponding 
        to the total number of actions as defined by
        the environment.
    atom_size : int 
        Number of atoms to use for the categorical DQN algorithm.
    support : torch.Tensor
        The support vector used in the distribution projection.
    activation : str (default: 'gelu')
        Activation function used in the block.
    num_hiddens : int (default: 128)
        Number of hidden units in the fully connected layer.
    enable_base_model : bool (default: False)
        Whether to use the base model or the ResNet.
    verbose : bool (default: False)
        Whether to print the model configuration.
    """

    def __init__(self, in_dim, out_dim, atom_size, support, num_hiddens=128, verbose=False):
        super(RainbowDQN, self).__init__()

        c, h, w = in_dim
        self.support = support
        self.out_dim = out_dim
        self.atom_size = atom_size

        if verbose:
            print(f"Configuring CNN backbone for input shape: ({c}, {h}, {w})")

        backbone = nn.Sequential(
            nn.Conv2d(c, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Calculate output size dynamically
        dummy_input = torch.zeros(1, c, h, w)
        dummy_output = backbone(dummy_input)
        output_size = dummy_output.size(1)

        # set common feature layer
        self.neural = nn.Sequential(
            nn.Conv2d(c, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(output_size, 512),
            nn.ReLU(),
            nn.Linear(512, num_hiddens)
        )

        # Initialize neural network
        self.neural.apply(lambda m: init(m, activation='relu'))

        # set advantage layer
        self.advantage_hidden_layer = NoisyLinear(
            num_hiddens, num_hiddens, 0.5)
        self.advantage_layer = NoisyLinear(
            num_hiddens, out_dim * atom_size, 0.5)

        # set value layer
        self.value_hidden_layer = NoisyLinear(num_hiddens, num_hiddens, 0.5)
        self.value_layer = NoisyLinear(num_hiddens, atom_size, 0.5)

        self.num_params = self.calc_num_params()
        if verbose:
            print(f"Total number of parameters: {self.num_params}")

    def forward(self, x, mask=None):
        """Forward propagation of input.

        Attributes
        ----------
        x : torch.Tensor
            Input tensor to be propagated to the model.
        mask : torch.Tensor (default: None)
            Masking tensor for invalid actions.
        """
        dist = self.dist(x)
        q = torch.sum(dist * self.support, dim=2)

        return q

    def dist(self, x):
        """Get distribution for atoms.

        Attributes
        ----------
        x : torch.Tensor
            Input tensor to be propagated to the model.
        """
        feature = self.neural(x)
        adv_hid = F.relu(self.advantage_hidden_layer(feature))
        val_hid = F.relu(self.value_hidden_layer(feature))

        advantage = self.advantage_layer(adv_hid).view(
            -1, self.out_dim, self.atom_size)
        value = self.value_layer(val_hid).view(
            -1, 1, self.atom_size)

        q_atoms = value + advantage - advantage.mean(dim=1, keepdim=True)

        dist = F.softmax(q_atoms, dim=-1)
        dist = dist.clamp(min=1e-3)

        return dist

    def reset_noise(self):
        """Reset all noisy layers.
        """
        self.advantage_hidden_layer.reset_noise()
        self.advantage_layer.reset_noise()
        self.value_hidden_layer.reset_noise()
        self.value_layer.reset_noise()

    def calc_num_params(self):
        """Calculate number of parameters in the model.
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    from thop import profile
    in_dim = (4, 22 * 8, 22 * 8)
    out_dim = 3
    atom_size = 51
    support = torch.linspace(-21, 20, atom_size)
    model = RainbowDQN(in_dim, out_dim, atom_size, support, verbose=True)

    x = torch.randn(1, 4, 22 * 8, 22 * 8)

    q = model(x)
    print(q.shape)
    print(q)
    print()

    dist = model.dist(x)
    print(dist.shape)
    print(dist)
    print()

    mask = torch.tensor([[0, 1, 0]], dtype=torch.bool)

    q_masked = model(x, mask)
    print(q_masked.shape)
    print(q_masked)
    print()

    macs, params = profile(model, inputs=(x, ))

    print(f"MACs: {macs}")
    print(f"Params: {params}")

    """
    Dimensionality analysis:
        - Input: (1, 4, 22 * 8, 22 * 8) ==> (1, 4, 176, 176)
        - Output: (1, 3)
        - Number of parameters: 11,249,856 (11,368,536)
        - MACs: 14,389,299,712

        - Input: (1, 4, 8 * 8, 8 * 8) ==> (1, 4, 64, 64)
        - Output: (1, 3)
        - Number of parameters: 11,249,856 (11,368,536)
        - MACs: 1,903,002,112
    """
