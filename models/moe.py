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

two_layer_router = False

#noisy router options
noisy_router = True
softplus_layer = False
nd_nosoftplus = False # no softplus optimization but layer exists

smallest_final_layer = False

deepseek_lb = True

K = 2  # Number of experts to use per token for top-k gating

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


# Define the Expert class
class Expert(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Expert, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)

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
                    noise_scalse = F.softplus(noise_magnitude)
                    # Add scaled noise to the clean logits
                    sampled_noise = noise_scale * sampled_noise

                # Add scaled noise to the clean logits
                noisy_logits = clean_logits + sampled_noise
                clean_gating_scores = F.softmax(clean_logits, dim=2)
            else:
                # No noise during inference
                noisy_logits = clean_logits
            ret = noisy_logits
        return F.softmax(ret, dim=2), ret, clean_gating_scores


# Define the Mixture of Experts Layer class
class MoELayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_experts):
        super(MoELayer, self).__init__()
        self.experts = nn.ModuleList([Expert(input_dim, hidden_dim, output_dim) for _ in range(num_experts)])
        self.gate = GatingNetwork(input_dim, num_experts)
        if deepseek_lb:
            self.expert_biases = nn.Parameter(torch.zeros(num_experts))

    def forward(self, x, num_experts_per_tok=K, epoch=None):
        # import pdb; pdb.set_trace()
        x_shape = x.shape
        # import pdb; pdb.set_trace()
        gating_scores, logits, clean_gating_scores = self.gate(x, epoch=epoch)
        if deepseek_lb:
            gating_scores_orig = gating_scores
            gating_scores += self.expert_biases
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
        gating_scores = F.normalize(gating_scores, p=1, dim=2)
        
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=1)
        
        if len(x_shape) == 4:
            eo_shape = expert_outputs.shape
            expert_outputs = expert_outputs.reshape((eo_shape[0], eo_shape[1], 
                                        eo_shape[2]*eo_shape[3], eo_shape[4]))

        expert_outputs = expert_outputs.transpose(1, 2)
        output = torch.einsum('bte,bteo->bto', gating_scores, expert_outputs)

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