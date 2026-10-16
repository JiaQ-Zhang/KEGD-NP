import torch
import torch.nn.functional as F
import math
import numpy as np
import torch.nn as nn
import torch.optim as optim


class KANLinear(torch.nn.Module):
    def __init__(
            self,
            in_features,
            out_features,
            grid_size=5,
            spline_order=3,
            scale_noise=0.1,
            scale_base=1.0,
            scale_spline=1.0,
            enable_standalone_scale_spline=True,
            base_activation=torch.nn.SiLU,
            grid_eps=0.02,
            grid_range=[-1, 1],
    ):
        super(KANLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order

        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = (
            (
                    torch.arange(-spline_order, grid_size + spline_order + 1) * h
                    + grid_range[0]
            )
                .expand(in_features, -1)
                .contiguous()
        )
        self.register_buffer("grid", grid)

        self.base_weight = torch.nn.Parameter(torch.Tensor(out_features, in_features))
        self.spline_weight = torch.nn.Parameter(
            torch.Tensor(out_features, in_features, grid_size + spline_order)
        )
        if enable_standalone_scale_spline:
            self.spline_scaler = torch.nn.Parameter(
                torch.Tensor(out_features, in_features)
            )

        self.scale_noise = scale_noise
        self.scale_base = scale_base
        self.scale_spline = scale_spline
        self.enable_standalone_scale_spline = enable_standalone_scale_spline
        self.base_activation = base_activation()
        self.grid_eps = grid_eps

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5) * self.scale_base)
        with torch.no_grad():
            noise = (
                    (
                            torch.rand(self.grid_size + 1, self.in_features, self.out_features)
                            - 1 / 2
                    )
                    * self.scale_noise
                    / self.grid_size
            )
            self.spline_weight.data.copy_(
                (self.scale_spline if not self.enable_standalone_scale_spline else 1.0)
                * self.curve2coeff(
                    self.grid.T[self.spline_order: -self.spline_order],
                    noise,
                )
            )
            if self.enable_standalone_scale_spline:
                # torch.nn.init.constant_(self.spline_scaler, self.scale_spline)
                torch.nn.init.kaiming_uniform_(self.spline_scaler, a=math.sqrt(5) * self.scale_spline)

    def b_splines(self, x: torch.Tensor):

        assert x.dim() == 2 and x.size(1) == self.in_features

        grid: torch.Tensor = (
            self.grid
        )
        x = x.unsqueeze(-1)
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
        for k in range(1, self.spline_order + 1):
            bases = (
                            (x - grid[:, : -(k + 1)])
                            / (grid[:, k:-1] - grid[:, : -(k + 1)])
                            * bases[:, :, :-1]
                    ) + (
                            (grid[:, k + 1:] - x)
                            / (grid[:, k + 1:] - grid[:, 1:(-k)])
                            * bases[:, :, 1:]
                    )

        assert bases.size() == (
            x.size(0),
            self.in_features,
            self.grid_size + self.spline_order,
        )
        return bases.contiguous()

    def curve2coeff(self, x: torch.Tensor, y: torch.Tensor):
        assert x.dim() == 2 and x.size(1) == self.in_features
        assert y.size() == (x.size(0), self.in_features, self.out_features)

        A = self.b_splines(x).transpose(
            0, 1
        )
        B = y.transpose(0, 1).to(A.device)
        solution = torch.linalg.lstsq(
            A, B
        ).solution
        result = solution.permute(
            2, 0, 1
        )

        assert result.size() == (
            self.out_features,
            self.in_features,
            self.grid_size + self.spline_order,
        )
        return result.contiguous()

    @property
    def scaled_spline_weight(self):
        return self.spline_weight * (
            self.spline_scaler.unsqueeze(-1)
            if self.enable_standalone_scale_spline
            else 1.0
        )

    def forward(self, x: torch.Tensor):
        assert x.dim() == 2 and x.size(1) == self.in_features

        base_output = F.linear(self.base_activation(x), self.base_weight)
        spline_output = F.linear(
            self.b_splines(x).view(x.size(0), -1),
            self.scaled_spline_weight.view(self.out_features, -1),
        )
        return base_output + spline_output

    @torch.no_grad()
    def update_grid(self, x: torch.Tensor, margin=0.01):
        assert x.dim() == 2 and x.size(1) == self.in_features
        batch = x.size(0)

        splines = self.b_splines(x)
        splines = splines.permute(1, 0, 2)
        orig_coeff = self.scaled_spline_weight
        orig_coeff = orig_coeff.permute(1, 2, 0)
        unreduced_spline_output = torch.bmm(splines, orig_coeff)
        unreduced_spline_output = unreduced_spline_output.permute(
            1, 0, 2
        )

        x_sorted = torch.sort(x, dim=0)[0]
        grid_adaptive = x_sorted[
            torch.linspace(
                0, batch - 1, self.grid_size + 1, dtype=torch.int64, device=x.device
            )
        ]

        uniform_step = (x_sorted[-1] - x_sorted[0] + 2 * margin) / self.grid_size
        grid_uniform = (
                torch.arange(
                    self.grid_size + 1, dtype=torch.float32, device=x.device
                ).unsqueeze(1)
                * uniform_step
                + x_sorted[0]
                - margin
        )

        grid = self.grid_eps * grid_uniform + (1 - self.grid_eps) * grid_adaptive
        grid = torch.concatenate(
            [
                grid[:1]
                - uniform_step
                * torch.arange(self.spline_order, 0, -1, device=x.device).unsqueeze(1),
                grid,
                grid[-1:]
                + uniform_step
                * torch.arange(1, self.spline_order + 1, device=x.device).unsqueeze(1),
            ],
            dim=0,
        )

        self.grid.copy_(grid.T)
        self.spline_weight.data.copy_(self.curve2coeff(x, unreduced_spline_output))

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):

        l1_fake = self.spline_weight.abs().mean(-1)
        regularization_loss_activation = l1_fake.sum()
        p = l1_fake / regularization_loss_activation
        regularization_loss_entropy = -torch.sum(p * p.log())
        return (
                regularize_activation * regularization_loss_activation
                + regularize_entropy * regularization_loss_entropy
        )


import torch
import torch.nn.functional as F
import math
import numpy as np


class SparseKAAGCNLayer(torch.nn.Module):
    def __init__(self, in_features, out_features, grid_size=5, spline_order=3, heads=1, concat=True):
        super(SparseKAAGCNLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.heads = heads
        self.concat = concat

        self.linear_out_features = self.heads * self.out_features  # 2*100=200
        self.linear = torch.nn.Linear(in_features, self.linear_out_features)

        self.self_param = torch.nn.Parameter(torch.Tensor(self.heads, self.out_features))

        self.attention_kan = KANLinear(
            in_features=2 * self.linear_out_features,
            out_features=heads,
            grid_size=grid_size,
            spline_order=spline_order
        )
        self.reset_parameters()



    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.linear.weight)
        torch.nn.init.zeros_(self.linear.bias)
        torch.nn.init.ones_(self.self_param)

    def forward(self, x, edge_index):
        h = self.linear(x)
        N = h.size(0)


        h = h.view(N, self.heads, self.out_features)

        row, col = edge_index
        valid = (row < N) & (row >= 0) & (col < N) & (col >= 0)
        row = row[valid]
        col = col[valid]

        E = row.size(0)
        x_i = h[row]
        x_j = h[col]


        linear_dim = self.heads * self.out_features
        x_i_flat = x_i.contiguous().view(E, linear_dim)
        x_j_flat = x_j.contiguous().view(E, linear_dim)

        pair_features = torch.cat([x_i_flat, x_j_flat], dim=1)

        attention_scores = self.attention_kan(pair_features)
        attention_scores = F.leaky_relu(attention_scores, negative_slope=0.2)
        attention_weights = self.sparse_softmax(attention_scores, (row, col), N)

        neighbor_agg = self.sparse_aggregation(attention_weights, h, (row, col))
        self_weight = self.self_param.view(1, self.heads, self.out_features)
        output = neighbor_agg + self_weight * h
        if self.concat and self.heads > 1:
            output = output.contiguous().view(-1, self.heads * self.out_features)
        else:
            output = output.mean(dim=1)

        return F.elu(output) if self.concat else output




    def sparse_softmax(self, attention_scores, edge_index, num_nodes):
        row, col = edge_index

        max_per_node = torch.zeros(num_nodes, self.heads,
                                   device=attention_scores.device)
        max_per_node.scatter_reduce_(0, col.unsqueeze(1).expand(-1, self.heads),
                                     attention_scores, reduce='amax', include_self=False)

        exp_scores = torch.exp(attention_scores - max_per_node[col])
        sum_exp = torch.zeros(num_nodes, self.heads, device=attention_scores.device)
        sum_exp.scatter_add_(0, col.unsqueeze(1).expand(-1, self.heads), exp_scores)

        attention_weights = exp_scores / (sum_exp[col] + 1e-8)

        return attention_weights

    def sparse_aggregation(self, attention_weights, h, edge_index):
        row, col = edge_index
        num_nodes = h.size(0)


        h = h.view(num_nodes, self.heads, self.out_features)

        source_features = h[row]

        weighted_features = attention_weights.unsqueeze(
            -1) * source_features

        aggregated = torch.zeros(num_nodes, self.heads, self.out_features,
                                 device=h.device)
        aggregated.scatter_add_(0, col.unsqueeze(1).unsqueeze(2).expand(-1, self.heads, self.out_features),
                                weighted_features)

        return aggregated


class ImprovedKAAGCN(torch.nn.Module):
    def __init__(self, N, dim, layer_hidden, layer_output, grid_size=5, spline_order=3, heads=1, dropout_rate=0.2):
        super(ImprovedKAAGCN, self).__init__()
        self.embed_fingerprint = nn.Embedding(N, dim)

        self.dropout = nn.Dropout(dropout_rate)


        self.layers = nn.ModuleList()
        for i in range(layer_hidden):
            in_dim = dim if i == 0 else (dim * heads if heads > 1 else dim)
            out_dim = dim
            self.layers.append(
                SparseKAAGCNLayer(in_dim, out_dim, grid_size, spline_order, heads)
            )

        output_dim = dim * heads if heads > 1 else dim
        self.W_output = nn.ModuleList([
            nn.Sequential(
                nn.Linear(output_dim, output_dim),
                nn.BatchNorm1d(output_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ) for _ in range(layer_output)
        ])
        self.W_property = nn.Linear(output_dim, 1)

        self.apply(self._init_weights)

    def dense_to_sparse(self, adj_matrix):
        N = adj_matrix.size(0)
        edge_index = adj_matrix.nonzero(as_tuple=False).t()


        valid = (edge_index[0] < N) & (edge_index[1] < N)
        edge_index = edge_index[:, valid]

        return edge_index



    def gnn(self, inputs):
        fingerprints = inputs[1]
        adjacencies = inputs[2]
        molecular_sizes = inputs[3]


        fingerprints_cat = torch.cat(fingerprints)
        fingerprint_vectors = self.embed_fingerprint(fingerprints_cat)
        fingerprint_vectors = self.dropout(fingerprint_vectors)

        adjacencies_cat = self.pad(adjacencies, 0)

        for i, layer in enumerate(self.layers):
            adj = adjacencies_cat
            fingerprint_vectors = layer.update(adj, fingerprint_vectors, i)
            if i < len(self.layers) - 1:
                fingerprint_vectors = self.dropout(fingerprint_vectors)

        molecular_vectors = self.sum(fingerprint_vectors, molecular_sizes)

        return None, molecular_vectors

    def pad(self, matrices, pad_value):

        shapes = [m.shape for m in matrices]
        M, N = sum([s[0] for s in shapes]), sum([s[1] for s in shapes])
        zeros = torch.zeros((M, N), device=matrices[0].device)
        pad_matrices = pad_value + zeros
        i, j = 0, 0
        for k, matrix in enumerate(matrices):
            m, n = matrix.shape
            pad_matrices[i:i + m, j:j + n] = matrix
            i += m
            j += n
        return pad_matrices



    def sum(self, vectors, axis):
        total_size = sum(axis)
        assert total_size == vectors.size(
            0), f"sum(molecular_sizes) = {total_size}, but vectors.size(0) = {vectors.size(0)}"
        sum_vectors = [torch.sum(v, 0) for v in torch.split(vectors, axis)]
        return torch.stack(sum_vectors)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                module.bias.data.zero_()

    def mlp(self, vectors):
        for layer in self.W_output:
            vectors = layer(vectors)
        outputs = self.W_property(vectors)
        return outputs

    def forward_regressor(self, data_batch, train):
        """前向传播 - 回归器"""
        inputs = data_batch[:-1]
        correct_values = torch.cat(data_batch[-1])

        if train:
            Smiles, molecular_vectors = self.gnn(inputs)
            predicted_values = self.mlp(molecular_vectors)
            loss = F.mse_loss(predicted_values, correct_values)
            return loss
        else:
            with torch.no_grad():
                Smiles, molecular_vectors = self.gnn(inputs)
                predicted_values = self.mlp(molecular_vectors)
            predicted_values = predicted_values.to('cpu').data.numpy()
            correct_values = correct_values.to('cpu').data.numpy()
            predicted_values = np.concatenate(predicted_values)
            correct_values = np.concatenate(correct_values)
            return Smiles, predicted_values, correct_values

    def forward_feature(self, inputs):

        _, molecular_vectors = self.gnn(inputs[:4])

        return molecular_vectors


def update(self, adj_matrix, vectors, layer_idx=0):
    edge_index = self.dense_to_sparse(adj_matrix)
    return self(vectors, edge_index)

SparseKAAGCNLayer.update = update
def dense_to_sparse(self, adj_tensor):

    edge_index = adj_tensor.nonzero(as_tuple=False).t()
    return edge_index
SparseKAAGCNLayer.dense_to_sparse = dense_to_sparse
