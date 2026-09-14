import math
import torch
import torch.nn as nn

class LocalGlobalNet(nn.Module):
    """Local-global network for hex grid value prediction.

    Expected input shapes/dtypes:
    - `local_features`: torch.FloatTensor, shape (B, N, K) where N = rows*columns, K = per-cell local input dim.
    - `global_context`: torch.FloatTensor, shape (B, G) where G == `global_context_dimension` (for example: goal one-hot of length N plus extra scalars such as number of sonars).

    Output:
    - predictions: torch.FloatTensor, shape (B, out_dim) where `out_dim` == `global_output_dimension`.

    Example:
        # build model from dataset dims
        model = LocalGlobalNet.from_dims(rows=20, columns=20, local_in=45, out_size=400)
    """

    def __init__(
        self,
        rows: int = 20,
        columns: int = 20,
        local_input_dimension: int = 45,
        local_hidden_dimension: int = 16, # originally 4
        local_output_dimension: int = 4, # originally 1
        global_context_dimension: int = 32,
        global_hidden_dimension: int = 128, # originally 32
        global_output_dimension: int = 1
    ):
        super().__init__()

        self.rows = rows
        self.columns = columns
        self.total_cells = self.rows * self.columns
        self.local_input_dimension = local_input_dimension
        self.local_hidden_dimension = local_hidden_dimension
        self.local_output_dimension = local_output_dimension
        self.global_context_dimension = global_context_dimension
        self.global_input_dimension = self.total_cells * self.local_output_dimension + self.global_context_dimension
        self.global_hidden_dimension = global_hidden_dimension
        self.global_output_dimension = global_output_dimension

        # per-cell local MLP parameters
        self.local_hidden_weight = nn.Parameter(torch.empty(self.total_cells, self.local_hidden_dimension, self.local_input_dimension))
        self.local_hidden_bias = nn.Parameter(torch.empty(self.total_cells, self.local_hidden_dimension))
        self.local_output_weight = nn.Parameter(torch.empty(self.total_cells, self.local_output_dimension, self.local_hidden_dimension))
        self.local_output_bias = nn.Parameter(torch.empty(self.total_cells, self.local_output_dimension))

        nn.init.kaiming_uniform_(self.local_hidden_weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.local_output_weight, a=math.sqrt(5))
        bound = 1.0 / math.sqrt(self.local_input_dimension)
        nn.init.uniform_(self.local_hidden_bias, -bound, bound)
        bound = 1.0 / math.sqrt(self.local_hidden_dimension)
        nn.init.uniform_(self.local_output_bias, -bound, bound)

        # global MLP
        self.global_hidden = nn.Linear(self.global_input_dimension, self.global_hidden_dimension)
        self.global_output = nn.Linear(self.global_hidden_dimension, self.global_output_dimension)

        nn.init.kaiming_uniform_(self.global_hidden.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.global_output.weight, a=math.sqrt(5))
        bound = 1.0 / math.sqrt(self.global_input_dimension)
        nn.init.uniform_(self.global_hidden.bias, -bound, bound)
        bound = 1.0 / math.sqrt(self.global_hidden_dimension)
        nn.init.uniform_(self.global_output.bias, -bound, bound)

    def forward(
        self,
        local_features: torch.Tensor,
        global_context: torch.Tensor,
    ) -> torch.Tensor:

        # runtime shape checks to fail early on mismatch
        if local_features.dim() != 3:
            raise ValueError(f"local_features must be 3D (B,N,K); got ndim={local_features.dim()}")
        if local_features.size(1) != self.total_cells:
            raise ValueError(f"local_features N dimension ({local_features.size(1)}) != expected total_cells ({self.total_cells})")
        if local_features.size(2) != self.local_input_dimension:
            raise ValueError(f"local_features K dimension ({local_features.size(2)}) != expected local_input_dimension ({self.local_input_dimension})")
        if global_context is None:
            raise ValueError("global_context must be provided (e.g. concatenated goal_onehot and extra scalars)")
        if global_context.dim() != 2:
            raise ValueError(f"global_context must be 2D (B,G); got ndim={global_context.dim()}")
        if global_context.size(1) != self.global_context_dimension:
            raise ValueError(f"global_context G dimension ({global_context.size(1)}) != expected global_context_dimension ({self.global_context_dimension})")

        batch_size = local_features.size(0)

        # Local network
        local_hidden = torch.einsum(
            "bnk,nhk->bnh",
            local_features,
            self.local_hidden_weight,
        )
        local_hidden += self.local_hidden_bias.unsqueeze(0)
        local_hidden = torch.relu(local_hidden)

        local_output = torch.einsum(
            "bnh,noh->bno",
            local_hidden,
            self.local_output_weight,
        )
        local_output += self.local_output_bias.unsqueeze(0)

        # Concatenate local and global information
        local_output = local_output.reshape(batch_size, -1)
        global_input = torch.cat([local_output, global_context], dim=1)

        # Global network
        global_hidden = torch.relu(self.global_hidden(global_input))

        prediction = self.global_output(global_hidden)

        return prediction