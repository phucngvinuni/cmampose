import math
import copy
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from timm.models.layers import trunc_normal_

# Standard 17-joint human skeleton topology (MM-Fi / COCO-like format)
edges_17 = torch.tensor([[0, 1], [1, 2], [2, 3],
                         [0, 4], [4, 5], [5, 6],
                         [0, 7], [7, 8], [8, 9], [9, 10],
                         [8, 11], [11, 12], [12, 13],
                         [8, 14], [14, 15], [15, 16]], dtype=torch.long)

# Try importing CUDA selective scan kernel, fall back to pure PyTorch if unavailable
try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
except ImportError:
    def selective_scan_fn(u, delta, A, B, C, D, z=None, delta_bias=None, delta_softplus=True, return_last_state=False):
        """ Pure PyTorch fallback implementation of Mamba selective scan (ZOH discretization) """
        val_b, val_d, val_l = u.shape
        val_n = A.shape[1]
        device = u.device
        
        if delta_bias is not None:
            delta = delta + delta_bias.unsqueeze(0).unsqueeze(-1)
        if delta_softplus:
            delta = F.softplus(delta)
            
        h = torch.zeros(val_b, val_d, val_n, device=device, dtype=u.dtype)
        ys = []
        
        # Pre-calculate delta_A: (B, D, N, L)
        delta_A = torch.exp(delta.unsqueeze(2) * A.unsqueeze(0).unsqueeze(-1))
        
        for t in range(val_l):
            delta_t = delta[:, :, t]
            u_t = u[:, :, t]
            B_t = B[:, :, t]
            C_t = C[:, :, t]
            
            bar_B_t = delta_t.unsqueeze(-1) * B_t.unsqueeze(1)
            delta_A_t = delta_A[:, :, :, t]
            h = delta_A_t * h + bar_B_t * u_t.unsqueeze(-1)
            
            y_t = (C_t.unsqueeze(1) * h).sum(dim=-1)
            ys.append(y_t)
            
        y = torch.stack(ys, dim=-1) # (B, D, L)
        
        if D is not None:
            y = y + u * D.unsqueeze(0).unsqueeze(-1)
            
        if z is not None:
            y = y * F.silu(z)
            
        return y


# ==========================================
# 1. Complex-Valued Neural Network Components
# ==========================================

class ComplexConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, dilation=1, stride=1):
        super().__init__()
        assert in_channels % 2 == 0
        assert out_channels % 2 == 0
        real_in = in_channels // 2
        real_out = out_channels // 2
        self.conv_r = nn.Conv2d(real_in, real_out, kernel_size, padding=padding, dilation=dilation, stride=stride, bias=False)
        self.conv_i = nn.Conv2d(real_in, real_out, kernel_size, padding=padding, dilation=dilation, stride=stride, bias=False)

    def forward(self, x):
        x_r, x_i = torch.chunk(x, 2, dim=1)
        out_r = self.conv_r(x_r) - self.conv_i(x_i)
        out_i = self.conv_r(x_i) + self.conv_i(x_r)
        return torch.cat([out_r, out_i], dim=1)


class ComplexModReLU(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.b = nn.Parameter(torch.zeros(channels // 2))

    def forward(self, x):
        x_r, x_i = torch.chunk(x, 2, dim=1)
        magnitude = torch.sqrt(x_r ** 2 + x_i ** 2 + 1e-8)
        b_unsqueezed = self.b.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        relu_mag = F.relu(magnitude + b_unsqueezed)
        ratio = relu_mag / magnitude
        out_r = x_r * ratio
        out_i = x_i * ratio
        return torch.cat([out_r, out_i], dim=1)


class ComplexDSKConv(nn.Module):
    def __init__(self, input_dim, output_dim, branches=2, reduction=4, min_dim=8):
        super().__init__()
        assert output_dim % 2 == 0
        hidden_dim = max(int(output_dim / reduction), min_dim)
        self.branches = branches
        self.output_dim = output_dim
        self.convs = nn.ModuleList([
            nn.Sequential(
                ComplexConv2d(input_dim, output_dim, kernel_size=3, padding=1 + i, dilation=1 + i),
                ComplexModReLU(output_dim),
            ) for i in range(branches)
        ])
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(output_dim, hidden_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.fcs = nn.ModuleList([nn.Conv2d(hidden_dim, output_dim // 2, kernel_size=1) for _ in range(branches)])
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        feats = torch.stack([conv(x) for conv in self.convs], dim=1)
        feats_u = feats.sum(dim=1)
        f_r, f_i = torch.chunk(feats_u, 2, dim=1)
        magnitude = torch.sqrt(f_r ** 2 + f_i ** 2 + 1e-8)
        magnitude_cat = torch.cat([magnitude, magnitude], dim=1)
        feats_s = self.gap(magnitude_cat)
        feats_z = self.fc(feats_s)
        attention = torch.stack([fc(feats_z) for fc in self.fcs], dim=1)
        attention = torch.cat([attention, attention], dim=2)
        attention = self.softmax(attention)
        return (feats * attention).sum(dim=1)


class DSKConv(nn.Module):
    """ Real-valued Dynamic Selective Receptive Field Convolution """
    def __init__(self, input_dim, output_dim, branches=2, groups=1, reduction=4, min_dim=8):
        super().__init__()
        hidden_dim = max(int(output_dim / reduction), min_dim)
        self.branches = branches
        self.output_dim = output_dim
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(input_dim, output_dim, kernel_size=3, stride=1, padding=1 + i, dilation=1 + i, groups=groups, bias=False),
                nn.BatchNorm2d(output_dim),
                nn.ReLU(inplace=True),
            ) for i in range(branches)
        ])
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(output_dim, hidden_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.fcs = nn.ModuleList([nn.Conv2d(hidden_dim, output_dim, kernel_size=1) for _ in range(branches)])
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        feats = torch.stack([conv(x) for conv in self.convs], dim=1)
        feats_u = feats.sum(dim=1)
        feats_s = self.gap(feats_u)
        feats_z = self.fc(feats_s)
        attention = torch.stack([fc(feats_z) for fc in self.fcs], dim=1)
        attention = self.softmax(attention)
        return (feats * attention).sum(dim=1)


class ComplexToMagnitude(nn.Module):
    def forward(self, x):
        dim = x.dim()
        if dim == 3:
            x_r, x_i = torch.chunk(x, 2, dim=-1)
        else:
            x_r, x_i = torch.chunk(x, 2, dim=1)
        return torch.sqrt(x_r ** 2 + x_i ** 2 + 1e-8)


class ComplexLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.fc_r = nn.Linear(in_features, out_features, bias=bias)
        self.fc_i = nn.Linear(in_features, out_features, bias=False)

    def forward(self, x):
        x_r, x_i = torch.chunk(x, 2, dim=-1)
        out_r = self.fc_r(x_r) - self.fc_i(x_i)
        out_i = self.fc_r(x_i) + self.fc_i(x_r)
        return torch.cat([out_r, out_i], dim=-1)


class ComplexConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, groups=1, bias=True):
        super().__init__()
        self.conv_r = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, groups=groups, bias=bias)
        self.conv_i = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, groups=groups, bias=False)

    def forward(self, x):
        x_r, x_i = torch.chunk(x, 2, dim=1)
        out_r = self.conv_r(x_r) - self.conv_i(x_i)
        out_i = self.conv_r(x_i) + self.conv_i(x_r)
        return torch.cat([out_r, out_i], dim=1)


class ComplexModReLUActivation(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.b = nn.Parameter(torch.zeros(channels))

    def forward(self, x):
        x_r, x_i = torch.chunk(x, 2, dim=-1)
        magnitude = torch.sqrt(x_r ** 2 + x_i ** 2 + 1e-8)
        b_unsqueezed = self.b.view(1, 1, -1)
        relu_mag = F.relu(magnitude + b_unsqueezed)
        ratio = relu_mag / magnitude
        out_r = x_r * ratio
        out_i = x_i * ratio
        return torch.cat([out_r, out_i], dim=-1)


class ComplexLayerNorm(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.ln_r = nn.LayerNorm(d_model)
        self.ln_i = nn.LayerNorm(d_model)

    def forward(self, x):
        x_r, x_i = torch.chunk(x, 2, dim=-1)
        out_r = self.ln_r(x_r)
        out_i = self.ln_i(x_i)
        return torch.cat([out_r, out_i], dim=-1)


class ComplexDropout(nn.Module):
    """ Drops entire channels uniformly across real and imaginary parts to preserve phase angles. """
    def __init__(self, p=0.1):
        super().__init__()
        self.p = p

    def forward(self, x):
        if not self.training or self.p == 0.0:
            return x
        x_r, x_i = torch.chunk(x, 2, dim=-1)
        mask = (torch.rand_like(x_r) >= self.p).float()
        mask = mask / (1.0 - self.p)
        return torch.cat([x_r * mask, x_i * mask], dim=-1)


# ==========================================
# 2. Spatiotemporal Complex Mamba Blocks
# ==========================================

class ComplexRoPEMamba(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, dropout=0.3):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(expand * d_model)
        self.dt_rank = max(1, int(d_model / 16))

        self.in_proj = ComplexLinear(d_model, self.d_inner * 2, bias=False)
        self.conv1d = ComplexConv1d(self.d_inner, self.d_inner, kernel_size=d_conv, groups=self.d_inner, padding=d_conv - 1)
        self.activation = ComplexModReLUActivation(self.d_inner)
        self.complex_dropout = ComplexDropout(dropout)
        
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + d_state * 2, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)
        
        a_init = repeat(torch.arange(1, d_state + 1, dtype=torch.float32), 'n -> d n', d=self.d_inner)
        self.A_log = nn.Parameter(torch.log(a_init))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        
        freqs = torch.exp(-torch.arange(0, d_state, 2).float() / d_state * math.log(10000.0))
        self.freqs = nn.Parameter(freqs)
        self.out_proj = ComplexLinear(self.d_inner, d_model, bias=False)

    def apply_rope_vectorized(self, x):
        batch_size, seq_len, _ = x.shape
        t = torch.arange(seq_len, device=x.device, dtype=self.freqs.dtype)
        freqs_t = torch.outer(t, self.freqs)
        cos_t = torch.cos(freqs_t).unsqueeze(0)
        sin_t = torch.sin(freqs_t).unsqueeze(0)
        
        x_real = x[:, :, 0::2]
        x_imag = x[:, :, 1::2]
        
        out_real = x_real * cos_t - x_imag * sin_t
        out_imag = x_real * sin_t + x_imag * cos_t
        
        out = torch.empty_like(x)
        out[:, :, 0::2] = out_real
        out[:, :, 1::2] = out_imag
        return out

    def forward(self, x):
        batch_size, sequence_length, _ = x.shape
        x_and_res = self.in_proj(x)
        x_and_res = self.complex_dropout(x_and_res)
        x_in, res = x_and_res.chunk(2, dim=-1)

        x_in = rearrange(x_in, 'b l d -> b d l')
        x_in = self.conv1d(x_in)[:, :, :sequence_length]
        x_in = rearrange(x_in, 'b d l -> b l d')
        x_in = self.activation(x_in)
        x_in = self.complex_dropout(x_in)

        x_in_r, x_in_i = torch.chunk(x_in, 2, dim=-1)
        x_in_mag = torch.sqrt(x_in_r ** 2 + x_in_i ** 2 + 1e-8)
        
        x_dbl = self.x_proj(x_in_mag)
        dt, b_param, c_param = x_dbl.split([self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = self.dt_proj(dt)

        if b_param.shape[1] > 1:
            b_cls = b_param[:, 0:1, :]
            b_local = b_param[:, 1:, :]
            b_rotated_local = self.apply_rope_vectorized(b_local)
            b_rotated = torch.cat([b_cls, b_rotated_local], dim=1)

            c_cls = c_param[:, 0:1, :]
            c_local = c_param[:, 1:, :]
            c_rotated_local = self.apply_rope_vectorized(c_local)
            c_rotated = torch.cat([c_cls, c_rotated_local], dim=1)
        else:
            b_rotated = self.apply_rope_vectorized(b_param)
            c_rotated = self.apply_rope_vectorized(c_param)

        u_r = rearrange(x_in_r, 'b l d -> b d l').contiguous()
        u_i = rearrange(x_in_i, 'b l d -> b d l').contiguous()
        u_stacked = torch.cat([u_r, u_i], dim=0)

        delta = rearrange(dt, 'b l d -> b d l').contiguous()
        delta_stacked = torch.cat([delta, delta], dim=0)

        A = -torch.exp(self.A_log.float()).contiguous()

        B = rearrange(b_rotated, 'b l d -> b d l').contiguous()
        B_stacked = torch.cat([B, B], dim=0)

        C = rearrange(c_rotated, 'b l d -> b d l').contiguous()
        C_stacked = torch.cat([C, C], dim=0)

        D = self.D.float().contiguous()

        res_r, res_i = torch.chunk(res, 2, dim=-1)
        z_r = rearrange(res_r, 'b l d -> b d l').contiguous()
        z_i = rearrange(res_i, 'b l d -> b d l').contiguous()
        z_stacked = torch.cat([z_r, z_i], dim=0)

        y_stacked = selective_scan_fn(
            u_stacked, delta_stacked, A, B_stacked, C_stacked, D, z=z_stacked,
            delta_bias=None,
            delta_softplus=True,
            return_last_state=False
        )

        y_stacked = rearrange(y_stacked, 'b d l -> b l d')
        y_r, y_i = torch.chunk(y_stacked, 2, dim=0)
        y = torch.cat([y_r, y_i], dim=-1)

        out = self.out_proj(y)
        out = self.complex_dropout(out)
        return out


class ComplexRoPEMambaBlock(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, dropout=0.3):
        super().__init__()
        self.norm = ComplexLayerNorm(d_model)
        self.mamba = ComplexRoPEMamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand, dropout=dropout)
        self.complex_dropout = ComplexDropout(dropout)

    def forward(self, x):
        norm_x = self.norm(x)
        out = self.mamba(norm_x)
        out = self.complex_dropout(out)
        return x + out


class OmniFrontEndLateFusionFullComplexRegularized(nn.Module):
    def __init__(
        self,
        image_size=(114, 10),
        patch_size=(2, 2),
        emb_dim=256,
        num_layer=2,
        dropout=0.3,
        patchify_dim=128,
    ) -> None:
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.emb_dim = emb_dim
        self.patchify_dim = patchify_dim

        patch_h, patch_w = patch_size
        num_patches = (image_size[0] // patch_h) * (image_size[1] // patch_w)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, emb_dim * 2))
        self.pos_embedding = nn.Parameter(torch.zeros(num_patches, 1, emb_dim * 2))

        # 1. Phase Branch (Complex-valued)
        self.phase_patchify = nn.Sequential(
            ComplexConv2d(6, patchify_dim, kernel_size=3, padding=1),
            ComplexModReLU(patchify_dim),
            ComplexDSKConv(input_dim=patchify_dim, output_dim=patchify_dim, branches=2, reduction=4),
            ComplexConv2d(patchify_dim, patchify_dim, kernel_size=patch_size, padding=0, stride=patch_size),
        )

        # 2. Amplitude Branch (Real-valued, projected to Complex space later)
        self.amp_patchify = nn.Sequential(
            nn.Conv2d(3, patchify_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(patchify_dim),
            nn.ReLU(inplace=True),
            DSKConv(input_dim=patchify_dim, output_dim=patchify_dim, branches=2, groups=1, reduction=4, min_dim=8),
            nn.Conv2d(patchify_dim, patchify_dim, kernel_size=patch_size, stride=patch_size),
        )

        # 3. Fusion Complex Linear Layer
        fused_in_dim = int(1.5 * patchify_dim)
        self.fusion = nn.Sequential(
            ComplexLinear(fused_in_dim, emb_dim),
            ComplexLayerNorm(emb_dim),
        )

        # 4. Complex RoPE Mamba Sequence Blocks with Dropout
        self.transformer = nn.Sequential(*[
            ComplexRoPEMambaBlock(d_model=emb_dim, d_state=16, d_conv=4, expand=2, dropout=dropout)
            for _ in range(num_layer)
        ])
        
        self.layer_norm = ComplexLayerNorm(emb_dim)
        self.to_magnitude = ComplexToMagnitude()

        self.init_weight()

    def init_weight(self):
        trunc_normal_(self.cls_token, std=0.02)
        trunc_normal_(self.pos_embedding, std=0.02)
        for module in self.phase_patchify.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        for module in self.amp_patchify.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def dynamic_amplitude_normalization(self, amp_input):
        static_background = amp_input.mean(dim=-1, keepdim=True)
        dynamic_amp = amp_input - static_background
        mu = dynamic_amp.mean(dim=(1, 2, 3), keepdim=True)
        sigma = dynamic_amp.std(dim=(1, 2, 3), keepdim=True)
        amp_norm = (dynamic_amp - mu) / (sigma + 1e-8)
        return amp_norm

    def _encode(self, phase_img, amp_img):
        amp_norm = self.dynamic_amplitude_normalization(amp_img)
        phase_patches = self.phase_patchify(phase_img)
        amp_patches = self.amp_patchify(amp_norm)
        
        amp_complex = torch.cat([amp_patches, torch.zeros_like(amp_patches)], dim=1)
        
        phase_r, phase_i = torch.chunk(phase_patches, 2, dim=1)
        amp_r, amp_i = torch.chunk(amp_complex, 2, dim=1)
        
        fused_r = torch.cat([phase_r, amp_r], dim=1)
        fused_i = torch.cat([phase_i, amp_i], dim=1)
        fused = torch.cat([fused_r, fused_i], dim=1)
        
        fused = rearrange(fused, 'b c h w -> b (h w) c')
        patches = self.fusion(fused)
        
        patches = rearrange(patches, 'b p c -> p b c')
        patches = patches + self.pos_embedding
        patches = torch.cat([self.cls_token.expand(-1, patches.shape[1], -1), patches], dim=0)
        
        features = rearrange(patches, 't b c -> b t c')
        features = self.layer_norm(self.transformer(features))
        
        features_magnitude = self.to_magnitude(features)
        return features_magnitude

    def forward(self, phase_img, amp_img):
        features = self._encode(phase_img, amp_img)
        return features[:, 1:, :].mean(1)


# ==========================================
# 3. GCN and Chebyshev Graph Convolution Decoders
# ==========================================

def normalize_adj(mx):
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def adj_mx_from_edges(num_pts, edges, sparse=False):
    edges = np.array(edges, dtype=np.int32)
    data, i, j = np.ones(edges.shape[0]), edges[:, 0], edges[:, 1]
    adj_mx = sp.coo_matrix((data, (i, j)), shape=(num_pts, num_pts), dtype=np.float32)
    adj_mx = adj_mx + adj_mx.T.multiply(adj_mx.T > adj_mx) - adj_mx.multiply(adj_mx.T > adj_mx)
    adj_mx = normalize_adj(adj_mx + sp.eye(adj_mx.shape[0]))
    return torch.tensor(adj_mx.todense(), dtype=torch.float)


class ChebConv(nn.Module):
    def __init__(self, in_c, out_c, K, bias=True, normalize=True):
        super(ChebConv, self).__init__()
        self.normalize = normalize
        self.weight = nn.Parameter(torch.Tensor(K + 1, 1, in_c, out_c))
        init.xavier_normal_(self.weight)

        if bias:
            self.bias = nn.Parameter(torch.Tensor(1, 1, out_c))
            init.zeros_(self.bias)
        else:
            self.register_parameter("bias", None)
        self.K = K + 1

    def forward(self, inputs, graph):
        L = ChebConv.get_laplacian(graph, self.normalize)
        mul_L = self.cheb_polynomial(L).unsqueeze(1)
        result = torch.matmul(mul_L, inputs)
        result = torch.matmul(result, self.weight)
        result = torch.sum(result, dim=0) + self.bias
        return result

    def cheb_polynomial(self, laplacian):
        N = laplacian.size(0)
        multi_order_laplacian = torch.zeros([self.K, N, N], device=laplacian.device, dtype=torch.float)
        multi_order_laplacian[0] = torch.eye(N, device=laplacian.device, dtype=torch.float)

        if self.K == 1:
            return multi_order_laplacian
        else:
            multi_order_laplacian[1] = laplacian
            if self.K == 2:
                return multi_order_laplacian
            else:
                for k in range(2, self.K):
                    multi_order_laplacian[k] = 2 * torch.mm(laplacian, multi_order_laplacian[k - 1]) - \
                                               multi_order_laplacian[k - 2]
        return multi_order_laplacian

    @staticmethod
    def get_laplacian(graph, normalize):
        if normalize:
            D = torch.diag(torch.sum(graph, dim=-1) ** (-1 / 2))
            L = torch.eye(graph.size(0), device=graph.device, dtype=graph.dtype) - torch.mm(torch.mm(D, graph), D)
        else:
            D = torch.diag(torch.sum(graph, dim=-1))
            L = D - graph
        return L


class _GraphConv(nn.Module):
    def __init__(self, input_dim, output_dim, p_dropout=None):
        super(_GraphConv, self).__init__()
        self.gconv = ChebConv(input_dim, output_dim, K=2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p_dropout) if p_dropout is not None else None

    def forward(self, x, adj):
        x = self.gconv(x, adj)
        if self.dropout is not None:
            x = self.dropout(self.relu(x))
        x = self.relu(x)
        return x


class _ResChebGC(nn.Module):
    def __init__(self, adj, input_dim, output_dim, hid_dim, p_dropout):
        super(_ResChebGC, self).__init__()
        self.adj = adj
        self.gconv1 = _GraphConv(input_dim, hid_dim, p_dropout)
        self.gconv2 = _GraphConv(hid_dim, output_dim, p_dropout)

    def forward(self, x):
        residual = x
        out = self.gconv1(x, self.adj)
        out = self.gconv2(out, self.adj)
        return residual + out


class SublayerConnection(nn.Module):
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = nn.LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))


class GraAttenLayer(nn.Module):
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(GraAttenLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = nn.ModuleList([
            SublayerConnection(size, dropout),
            SublayerConnection(size, dropout)
        ])

    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)


def attention(Q, K, V, mask=None, dropout=None):
    d_k = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, V), p_attn


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.linears = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(4)])
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)
        Q, K, V = [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
                   for l, x in zip(self.linears, (query, key, value))]
        x, _ = attention(Q, K, V, mask=mask, dropout=self.dropout)
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)


class LAM_Gconv(nn.Module):
    def __init__(self, in_features, out_features, activation=nn.ReLU(inplace=True)):
        super(LAM_Gconv, self).__init__()
        self.fc = nn.Linear(in_features=in_features, out_features=out_features)
        self.activation = activation

    def laplacian_batch(self, A_hat):
        batch, N = A_hat.shape[:2]
        D_hat = (torch.sum(A_hat, 1) + 1e-5) ** (-0.5)
        L = D_hat.view(batch, N, 1) * A_hat * D_hat.view(batch, 1, N)
        return L

    def forward(self, X, A):
        batch = X.size(0)
        A_hat = A.unsqueeze(0).repeat(batch, 1, 1)
        X = self.fc(torch.bmm(self.laplacian_batch(A_hat), X))
        if self.activation is not None:
            X = self.activation(X)
        return X


class GraphNet(nn.Module):
    def __init__(self, in_features=2, out_features=2, n_pts=21):
        super(GraphNet, self).__init__()
        self.A_hat = nn.Parameter(torch.eye(n_pts).float(), requires_grad=True)
        self.gconv1 = LAM_Gconv(in_features, in_features * 2)
        self.gconv2 = LAM_Gconv(in_features * 2, out_features, activation=None)

    def forward(self, X):
        X_0 = self.gconv1(X, self.A_hat)
        X_1 = self.gconv2(X_0, self.A_hat)
        return X_1


class GraFormer_Strong_Dropout(nn.Module):
    def __init__(self, adj, hid_dim=128, in_feat=256, out_dim=3, num_layers=4,
                 n_head=4, dropout=0.4, n_pts=17):
        super().__init__()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.n_layers = num_layers
        self.adj = adj

        self.gconv_input = ChebConv(in_c=in_feat, out_c=hid_dim, K=2)
        _gconv_layers = []
        _attention_layer = []

        dim_model = hid_dim
        attn = MultiHeadedAttention(n_head, dim_model)
        gcn = GraphNet(in_features=dim_model, out_features=dim_model, n_pts=n_pts)

        self.src_mask = torch.tensor([[[True]*n_pts]]).to(device)

        for _ in range(num_layers):
            _gconv_layers.append(_ResChebGC(
                adj=self.adj, input_dim=hid_dim, output_dim=hid_dim,
                hid_dim=hid_dim, p_dropout=dropout
            ))
            _attention_layer.append(GraAttenLayer(dim_model, copy.deepcopy(attn), copy.deepcopy(gcn), dropout))

        self.gconv_layers = nn.ModuleList(_gconv_layers)
        self.atten_layers = nn.ModuleList(_attention_layer)
        self.gconv_output = ChebConv(in_c=dim_model, out_c=out_dim, K=2)

    def forward(self, x):
        out = self.gconv_input(x, self.adj)
        for i in range(self.n_layers):
            out = self.atten_layers[i](out, self.src_mask)
            out = self.gconv_layers[i](out)
        out = self.gconv_output(out, self.adj)
        return out


class GraFormer_Decoder_DynamicWithLateFusionFullComplexStrongDropout(nn.Module):
    def __init__(self, encoder, emb_dim=256, num_layers=4, n_head=4, dropout=0.4, graformer_hid_dim=128):
        super().__init__()
        self.encoder = encoder
        self.cross_attn = nn.MultiheadAttention(embed_dim=emb_dim, num_heads=4, batch_first=True, dropout=dropout)
        self.cross_attn_dropout = nn.Dropout(dropout)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        adj = adj_mx_from_edges(num_pts=17, edges=edges_17, sparse=False).to(device)
        self.graformer = GraFormer_Strong_Dropout(
            adj=adj,
            in_feat=emb_dim,
            hid_dim=graformer_hid_dim,
            out_dim=3,
            num_layers=num_layers,
            n_head=n_head,
            dropout=dropout,
            n_pts=17,
        )
        self.register_buffer('adj_buffer', adj)
        self.pose_query = nn.Parameter(torch.randn(1, 17, emb_dim))

    def forward(self, phase_img, amp_img):
        features = self.encoder._encode(phase_img, amp_img)
        features = features.permute(1, 0, 2) # [Seq_Len, B, emb_dim]

        memory = features[1:].permute(1, 0, 2) # [B, Seq_Len - 1, emb_dim]
        batch_size = memory.shape[0]

        query = self.pose_query.expand(batch_size, -1, -1)
        z_dynamic, _ = self.cross_attn(query=query, key=memory, value=memory)
        z_dynamic = self.cross_attn_dropout(z_dynamic)

        z = z_dynamic + query

        # Synchronize adjacency matrix
        self.graformer.adj = self.adj_buffer
        for layer in self.graformer.gconv_layers:
            layer.adj = self.adj_buffer
        self.graformer.src_mask = self.graformer.src_mask.to(z.device)

        pred = self.graformer(z)
        return pred, z


# ==========================================
# 4. CMambaPose SOTA Model Wrapper Class
# ==========================================

class CMambaPose(nn.Module):
    """
    C-MambaPose: A Physics-Informed Complex Mamba-GraFormer Hybrid Framework for robust
    cross-environment WiFi human pose estimation.
    """
    def __init__(
        self,
        image_size=(114, 10),
        patch_size=(2, 2),
        emb_dim=256,
        front_layer=2,
        front_head=4,
        input_dim=6,
        keypoints=17,
        coor_num=3,
        token_num=285,
        dataset='mmfi-csi',
        num_person=1,
        graformer_layers=4,
        graformer_heads=4,
        graformer_dropout=0.4,
        dropout=0.3,
        graformer_hid_dim=128,
        patchify_dim=128,
    ) -> None:
        super().__init__()
        encoder = OmniFrontEndLateFusionFullComplexRegularized(
            image_size=image_size,
            patch_size=patch_size,
            emb_dim=emb_dim,
            num_layer=front_layer,
            dropout=dropout,
            patchify_dim=patchify_dim,
        )

        self.decoder = GraFormer_Decoder_DynamicWithLateFusionFullComplexStrongDropout(
            encoder,
            emb_dim=emb_dim,
            num_layers=graformer_layers,
            n_head=graformer_heads,
            dropout=graformer_dropout,
            graformer_hid_dim=graformer_hid_dim,
        )

    def forward(self, phase_img, amp_img):
        return self.decoder(phase_img, amp_img)
