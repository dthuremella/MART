"""
This model integrates the MoE concept within a Transformer architecture. Each token's
representation is processed by a subset of experts, determined by the gating mechanism.
This architecture allows for efficient and specialized handling of different aspects of the
data, aiming for the adaptability and efficiency noted in the Mixtral 8x7B model's design
philosophy. The model activates only a fraction of the available experts for each token,
significantly reducing the computational resources needed compared to activating all experts
for all tokens.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

nomoe = True

cls_head = False

two_layer_router = False

one_router_same_expert = False

linearnet1e = False

kalman = False

#noisy router options (only one should be true)
noisy_router, softplus_layer, nd_nosoftplus = False, False, False # nd_nosoftplus: no softplus optimization but layer exists
gumbel_sigmoid, gumbel_softmax, tau_cd, tau_ed = False, True, False, False

even_harmonic_noop = [0, 1/14, 1/12, 1/10, 1/8, 1/6, 1/4, 1/2] # even harmonic from 0 to 1/2
no_op_double = [0, 0, 1/8, 1/8, 1/4, 1/4, 1/2, 1/2] 
no_op = even_harmonic_noop                  # None or even_harmonic or just 0, 1/2, 1/4, 1/8 twice

smallest_final_layer = False

deepseek_lb = False # imported by main function
biased_deepseek = [1/2, 1/4, 1/6, 1/8, 1/10, 1/12, 1/14, 0] # or None # even harmonic from 1/2 to 1/14, 0

K = 2  # Number of experts to use per token for top-k gating

NUM_EXPERTS = 8  # Total number of experts in the MoE layer

test_calculate_all_expert_outputs = False # True for baseline and fastest method


def _gumbel_sigmoid(logits, tau=1, training=True):
    if not training:
        return logits.sigmoid()
    if training:
        # ~Gumbel(0,1)
        gumbels1 = (
            -torch.empty_like(logits, memory_format=torch.legacy_contiguous_format)
            .exponential_()
            .log()
        )
        gumbels2 = (
            -torch.empty_like(logits, memory_format=torch.legacy_contiguous_format)
            .exponential_()
            .log()
        )
        # Difference of two gumbels because we apply a sigmoid
        gumbels1 = (logits + gumbels1 - gumbels2) / tau
        return gumbels1.sigmoid()

def cosine_decay(step: int, max_steps: int, max_amplitude: float, min_amplitude: float = 0.0):
    """
    Cosine decay schedule for noise amplitude (or any scalar).
    
    Args:
        step (int): Current step (0-based).
        max_steps (int): Total number of steps for full decay.
        max_amplitude (float): Starting value at step=0.
        min_amplitude (float): Final value at step=max_steps.
        
    Returns:
        float: Current amplitude.
    """
    step = min(step, max_steps)  # clamp so we don’t overshoot
    cos_decay = 0.5 * (1 + math.cos(math.pi * step / max_steps))
    return min_amplitude + (max_amplitude - min_amplitude) * cos_decay


# Define the Expert classes
class Expert(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Expert, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)

class NoOpExpert(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(NoOpExpert, self).__init__()
        # if linearnet1e:
        #     self.fc = nn.Linear(input_dim, output_dim)
    def forward(self, x):
        # if linearnet1e:
        #     return self.fc(x)
        # else:
        #     return x
        return x

# Define the Gating Network class
class GatingNetwork(nn.Module):
    def __init__(self, input_dim, num_experts, noise_stddev=1.0):
        super(GatingNetwork, self).__init__()
        if two_layer_router:
            self.gate_l1 = nn.Linear(input_dim, input_dim)
            self.gate_l2 = nn.Linear(input_dim, num_experts)
        else:
            self.gate = nn.Linear(input_dim, num_experts)
        
        if noisy_router:
            self.noise_stddev = noise_stddev # TODO reduce if it doesn't learn anything at beginning (too high)
            # TODO NORMALIZE (BATCH NORM) BEFOREHAND, THEN WEKNOW WHAT THE STD DEV SHOULD BE (AFTER A FEW STEPS, NORM WILL BE 1)
            if softplus_layer or nd_nosoftplus:
                self.noise_layer = nn.Linear(input_dim, num_experts, bias=False)

    def forward(self, x, epoch=None):
        if two_layer_router:
            x = F.relu(self.gate_l1(x))
            ret = self.gate_l2(x)
        else:
            ret = self.gate(x)

        clean_gating_scores = None
        if noisy_router:
            clean_logits = ret
            if self.training:
                ### cosine decay
                sampled_noise = torch.randn_like(clean_logits) * self.noise_stddev * cosine_decay(epoch, 300, 1.0, 0.0)
                
                if softplus_layer:
                    # We use a separate linear layer for noise magnitude, scaled by standard normal noise
                    # Shape: (B * S, num_experts)
                    noise_magnitude = self.noise_layer(x)
                    # Softplus ensures the magnitude scaling is positive
                    noise_scale = F.softplus(noise_magnitude)
                    # Add scaled noise to the clean logits
                    sampled_noise = noise_scale * sampled_noise

                # Add scaled noise to the clean logits
                noisy_logits = clean_logits + sampled_noise
                clean_gating_scores = F.softmax(clean_logits, dim=2)
            else:
                # No noise during inference
                noisy_logits = clean_logits
            ret = noisy_logits
        
        # apply gumbel noise
        tau = 1
        if tau_cd and self.training: tau = cosine_decay(epoch, 300, 1, 0)
        if tau_ed and self.training: tau = 10/(epoch+1)**0.5 
        if gumbel_sigmoid:
            return _gumbel_sigmoid(ret, tau=tau, training=self.training), ret, clean_gating_scores
        elif gumbel_softmax:
            return F.gumbel_softmax(ret, tau=tau, hard=(not self.training)), ret, clean_gating_scores

        return F.softmax(ret, dim=2), ret, clean_gating_scores


# Define the Mixture of Experts Layer class
class MoELayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_experts=NUM_EXPERTS):
        super(MoELayer, self).__init__()
        if no_op is not None:
            self.experts = []
            for i in range(num_experts):
                if no_op[i] == 0:
                    self.experts.append(NoOpExpert(input_dim, output_dim))
                else:
                    h = int(hidden_dim * no_op[i]) # even harmonic 1/14 to 1/2
                    self.experts.append(Expert(input_dim, h, output_dim))
            self.experts = nn.ModuleList(self.experts)
        else:
            self.experts = nn.ModuleList([Expert(input_dim, hidden_dim, output_dim) for _ in range(num_experts)])
        self.gate = GatingNetwork(input_dim, num_experts)
        if deepseek_lb:
            if biased_deepseek is None:
                self.expert_biases = nn.Parameter(torch.zeros(num_experts))
            else:
                self.expert_biases = nn.Parameter(torch.tensor(biased_deepseek, dtype=torch.float, device='cuda'))


    def forward(self, x, num_experts_per_tok=K, epoch=None, prev_gating_scores=None):
        x_shape = x.shape

        if one_router_same_expert and prev_gating_scores is not None:
            gating_scores = prev_gating_scores
            logits = None
        else:
            if cls_head:
                # use only cls token for gating
                if len(x_shape) == 4: gating_input = x[:,:1, :1]
                else: gating_input = x[:,:1]
            else:
                gating_input = x
            gating_scores, logits, clean_gating_scores = self.gate(gating_input, epoch=epoch)
            if deepseek_lb:
                gating_scores_orig = gating_scores
                gating_scores = gating_scores + self.expert_biases
            if len(x_shape) == 4:
                g_shape = gating_scores.shape
                gating_scores = gating_scores.reshape((g_shape[0], 
                                            g_shape[1]*g_shape[2], g_shape[3]))
                                            
            topk_gating_scores, topk_indices = gating_scores.topk(num_experts_per_tok, dim=2, sorted=False)
            # Create a mask to zero out the contributions of non-topk experts
            mask = torch.zeros_like(gating_scores).scatter_(2, topk_indices, 1) # TODO what does scatter do?  
            # Use the mask to retain only the topk gating scores
            if deepseek_lb:
                if len(x_shape) == 4:
                    g_shape = gating_scores_orig.shape
                    gating_scores_orig = gating_scores_orig.reshape((g_shape[0], 
                                                g_shape[1]*g_shape[2], g_shape[3]))
                gating_scores = gating_scores_orig * mask
            else:
                gating_scores = gating_scores * mask 
            # Normalize the gating scores to sum to 1 across the selected top experts
            gating_scores = F.normalize(gating_scores, p=1, dim=2) # [batch_size, num_tokens, num_experts]

        if self.training or test_calculate_all_expert_outputs:
            expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=1) # [batch_size, num_experts, num_tokens, output_dim]
            if len(x_shape) == 4:
                eo_shape = expert_outputs.shape
                expert_outputs = expert_outputs.reshape((eo_shape[0], eo_shape[1], 
                                            eo_shape[2]*eo_shape[3], eo_shape[4]))

        else:
            if len(x_shape) == 4:
                x = x.reshape((x_shape[0], x_shape[1]*x_shape[2], x_shape[3]))
            
            if x.shape[1] < NUM_EXPERTS:
                # slow code ######
                expert_outputs = torch.zeros((gating_scores.shape[0], gating_scores.shape[1], NUM_EXPERTS, x.shape[-1]), device=x.device) # [batch_size, num_tokens, num_experts, output_dim]
                for example_idx in range(expert_outputs.shape[0]): # batch size is 1 for testing
                    for token_idx in range(expert_outputs.shape[1]):
                        for expert_idx in torch.nonzero(gating_scores[example_idx, token_idx]):
                            expert_idx = expert_idx.item()
                            expert_output = self.experts[expert_idx](x[example_idx, token_idx, :])
                            expert_outputs[example_idx, token_idx, expert_idx] = expert_output
                expert_outputs = expert_outputs.transpose(1, 2) # [batch_size, num_tokens, num_experts, output_dim]

            else:
                batch_size, num_tokens, _ = gating_scores.shape
                # Initialize output (assuming experts output same dimension)
                # Adjust output shape based on your expert's output
                expert_outputs = torch.zeros(batch_size, NUM_EXPERTS, x.shape[1], x.shape[2], device=x.device)  # [batch_size, num_tokens, output_dim]
                # Process each expert separately
                for expert_idx in range(NUM_EXPERTS):
                    # Find which batch elements go to this expert
                    mask = (topk_indices == expert_idx)
                    batch_indices = torch.where(mask)[0]
                    
                    if len(batch_indices) > 0:
                        # Extract tokens for this expert
                        expert_input = x[batch_indices]  # Shape: [num_tokens_for_expert, 12, 64]
                        
                        # Process through the expert
                        expert_output = self.experts[expert_idx](expert_input)
                        
                        # Place results back in correct positions
                        expert_outputs[batch_indices, expert_idx] = expert_output
                # # During inference, only compute outputs for the top-k experts
                # expert_outputs_list = []
                # for i in range(NUM_EXPERTS):
                #     # Create a mask for tokens that selected expert i
                #     expert_mask = (topk_indices == i).any(dim=2).float().unsqueeze(-1)  # [batch_size, num_tokens, 1]
                #     if expert_mask.sum() > 0:
                #         if 1 in topk_indices:
                #             import pdb; pdb.set_trace()
                #         masked_x = x * expert_mask
                #         # get indices where expert_mask is non-zero
                #         # put non-zero values through expert
                #         # replace sparse tensor values with correct expert outputs
                #         expert_output = self.experts[i](masked_x)  # [batch_size, num_tokens, output_dim] ##### TODO: mask x, pass it through, view it back
                #     else:
                #         expert_output = torch.zeros((batch_size, num_tokens, x.shape[-1]), device=x.device)
                #     expert_outputs_list.append(expert_output)
                # expert_outputs = torch.stack(expert_outputs_list, dim=1)  # [batch_size, num_experts, num_tokens, output_dim]
        
        expert_outputs = expert_outputs.transpose(1, 2) # [batch_size, num_tokens, num_experts, output_dim]
        output = torch.einsum('bte,bteo->bto', gating_scores, expert_outputs) # [batch_size, num_tokens, output_dim]

        if len(x_shape) == 4:
            o_shape = output.shape
            output = output.reshape((o_shape[0], x_shape[1], x_shape[2], o_shape[-1]))

        if noisy_router and self.training:
            if len(x_shape) == 4:
                g_shape = clean_gating_scores.shape
                clean_gating_scores = clean_gating_scores.reshape((g_shape[0], 
                                            g_shape[1]*g_shape[2], g_shape[3]))
                                            
            topk_gating_scores, topk_indices = clean_gating_scores.topk(num_experts_per_tok, dim=2, sorted=False)
            # Create a mask to zero out the contributions of non-topk experts
            mask = torch.zeros_like(clean_gating_scores).scatter_(2, topk_indices, 1) # TODO what does scatter do?  
            # Use the mask to retain only the topk gating scores
            clean_gating_scores = clean_gating_scores * mask 
            # Normalize the gating scores to sum to 1 across the selected top experts
            clean_gating_scores = F.normalize(gating_scores, p=1, dim=2)            
            return output, clean_gating_scores, logits
        else: 
            return output, gating_scores, logits

# Define the overall Transformer model with integrated MoE
class TransformerWithMoE(nn.Module):
    def __init__(self, num_layers, dim, head_dim, hidden_dim, n_heads, num_experts, vocab_size, num_experts_per_tok):
        super(TransformerWithMoE, self).__init__()
        self.num_experts_per_tok = num_experts_per_tok
        self.embedding = nn.Embedding(vocab_size, dim)
        self.layers = nn.ModuleList([nn.TransformerEncoderLayer(d_model=dim, nhead=n_heads) for _ in range(num_layers)])
        self.moe_layer = MoELayer(dim, hidden_dim, dim, num_experts)
        self.output_layer = nn.Linear(dim, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        for layer in self.layers:
            x = layer(x)
        x = self.moe_layer(x, self.num_experts_per_tok)
        logits = self.output_layer(x)
        return logits

# Initialize the model with configurations matching Mixtral 8x7B
model = TransformerWithMoE(
    num_layers=32,              # Number of transformer layers
    dim=4096,                   # Dimension of the model
    head_dim=128,               # Dimension of each head in the multi-head attention mechanisms
    hidden_dim=14336,           # Hidden dimensionality in the feed-forward network within the transformer
    n_heads=32,                 # Number of attention heads
    num_experts=8,              # Number of experts in the MoE layer
    vocab_size=32000,           # Vocabulary size for the embedding layer
    num_experts_per_tok=2       # Number of experts activated per token
)

    
"""
Batched Kalman filter that exactly matches the original single-sample version,
extended to predict multiple future steps.

:param history: Tensor of shape (N, M, 2)
:param prediction_horizon: int — number of future steps to predict (C)
:return: Tensor of shape (N, C, 2)
"""
def kalman_filter(history, prediction_horizon):
    assert history.ndim == 3 and history.shape[2] == 2, \
        "history must be of shape (N, M, 2)"

    N, M, _ = history.shape

    z_x = history[:, :, 0].clone()
    z_y = history[:, :, 1].clone()

    # --- Handle NaNs exactly as original ---
    nan_mask = torch.isnan(z_x) | torch.isnan(z_y)
    num_nans = nan_mask.sum(dim=1)  # (N,)

    # Replace NaNs with zeros to avoid NaN propagation
    z_x = torch.nan_to_num(z_x, nan=0.0)
    z_y = torch.nan_to_num(z_y, nan=0.0)

    # Compute velocities exactly like original
    v_x = torch.zeros(N, device=history.device)
    v_y = torch.zeros(N, device=history.device)

    for i in range(M - 1):
        valid = ~nan_mask[:, i]
        v_x += valid * (z_x[:, i + 1] - z_x[:, i])
        v_y += valid * (z_y[:, i + 1] - z_y[:, i])

    effective_length = (M - num_nans).clamp_min(2)  # avoid divide-by-zero
    v_x /= (effective_length - 1)
    v_y /= (effective_length - 1)

    # Truncate leading NaNs
    min_nan = num_nans.min().item()
    z_x = z_x[:, min_nan:]
    z_y = z_y[:, min_nan:]
    length_history = M - min_nan

    # --- Allocate state ---
    x_x = torch.zeros((N, length_history + 1), device=history.device)
    x_y = torch.zeros((N, length_history + 1), device=history.device)
    P_x = torch.zeros_like(x_x)
    P_y = torch.zeros_like(x_y)
    P_vx = torch.zeros_like(x_x)
    P_vy = torch.zeros_like(x_y)

    P_x[:, 0] = 1.0
    P_y[:, 0] = 1.0
    P_vx[:, 0] = 1.0
    P_vy[:, 0] = 1.0
    x_x[:, 0] = z_x[:, 0]
    x_y[:, 0] = z_y[:, 0]

    Q = 0.00001
    R = 0.0001

    # --- Run Kalman filter exactly like original ---
    for k in range(length_history - 1):
        x_x[:, k + 1] = x_x[:, k] + v_x
        x_y[:, k + 1] = x_y[:, k] + v_y
        P_x[:, k + 1] = P_x[:, k] + P_vx[:, k] + Q
        P_y[:, k + 1] = P_y[:, k] + P_vy[:, k] + Q
        P_vx[:, k + 1] = P_vx[:, k] + Q
        P_vy[:, k + 1] = P_vy[:, k] + Q

        K_x = P_x[:, k + 1] / (P_x[:, k + 1] + R)
        K_y = P_y[:, k + 1] / (P_y[:, k + 1] + R)

        x_x[:, k + 1] = x_x[:, k + 1] + K_x * (z_x[:, k + 1] - x_x[:, k + 1])
        x_y[:, k + 1] = x_y[:, k + 1] + K_y * (z_y[:, k + 1] - x_y[:, k + 1])

        P_x[:, k + 1] = P_x[:, k + 1] - K_x * P_x[:, k + 1]
        P_y[:, k + 1] = P_y[:, k + 1] - K_y * P_y[:, k + 1]

        K_vx = P_vx[:, k + 1] / (P_vx[:, k + 1] + R)
        K_vy = P_vy[:, k + 1] / (P_vy[:, k + 1] + R)
        P_vx[:, k + 1] = P_vx[:, k + 1] - K_vx * P_vx[:, k + 1]
        P_vy[:, k + 1] = P_vy[:, k + 1] - K_vy * P_vy[:, k + 1]

    # --- Multi-step future prediction (C steps ahead) ---
    k = length_history - 1
    steps = torch.arange(1, prediction_horizon + 1, device=history.device).float().view(1, -1)

    x_future = x_x[:, k].unsqueeze(1) + v_x.unsqueeze(1) * steps
    y_future = x_y[:, k].unsqueeze(1) + v_y.unsqueeze(1) * steps

    predictions = torch.stack([x_future, y_future], dim=-1)  # (N, C, 2)
    return predictions

def kalman_score(data, obs_len, pred_len):
    kalman_preds = kalman_filter(data[:, :obs_len, :], pred_len)
    error = torch.norm(kalman_preds - data[:, obs_len:obs_len+pred_len, :], dim=-1)  # (N, C)
    return error.mean(dim=1)  # (N,) # ADE (could also do FDE by taking error[:, -1])