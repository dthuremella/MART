
import random
import tree

import yaml
import torch
import numpy as np

from box import Box

from fmoe.gates import NaiveGate
from fmoe.transformer import FMoE, _Expert, FMoETransformerMLP
from fmoe.layers import _fmoe_general_global_forward, fmoe_faster_schedule
from fmoe.functions import AllGather, Slice
import torch
import torch.nn.functional as F
from torch import nn

viz = True

moe_e = True
moe_n = True
even_harmonic = False
class_token = False # only done for second two layers, need to use even_harmonic for cls layers currently (don't turn on without even_harmonic)
class_token_all_layers = False
class_token_harmonic_div_is_always_1 = False

_call_count = 0
class _ExpertPrint(_Expert):
    def forward(self, inp, fwd_expert_count):
        global _call_count
        _call_count += 1
        # print(f"Expert call #{_call_count}, inp.shape: {inp.shape}")
        x = self.htoh4(inp, fwd_expert_count)
        x = self.activation(x)
        x = self.h4toh(x, fwd_expert_count)
        return x

class GSoftmaxGate(NaiveGate):
    r"""
    A gate that uses gumbel softmax to calculate the score of each expert.
    """

    def forward(self, inp, return_all_scores=False):
        r"""
        The naive implementation simply calculates the top-k of a linear layer's
        output.
        """
        gate = self.gate(inp)
        gate_top_k_val, gate_top_k_idx = torch.topk(
            gate, k=self.top_k, dim=-1, largest=True, sorted=False
        )  # [.. x top_k]
        gate_top_k_val = gate_top_k_val.view(-1, self.top_k)

        gate_score = F.gumbel_softmax(gate_top_k_val, tau=1, hard=(not self.training))

        # dummy loss
        self.set_loss(torch.zeros(1, requires_grad=True).to(inp.device))

        if return_all_scores:
            return gate_top_k_idx, gate_score, gate
        return gate_top_k_idx, gate_score

class GSoftmaxHarmonicGate(NaiveGate):
    r"""
    A gate that uses gumbel softmax to calculate the score of each expert.
    """

    def forward(self, inp, return_all_scores=False):
        r"""
        The naive implementation simply calculates the top-k of a linear layer's
        output.
        """
        gate = self.gate(inp)
        gate_top_k_val, gate_top_k_idx = torch.topk(
            gate, k=self.top_k, dim=-1, largest=True, sorted=False
        )  # [.. x top_k]
        gate_top_k_val = gate_top_k_val.view(-1, self.top_k)

        gate_score = F.gumbel_softmax(gate_top_k_val, tau=1, hard=(not self.training))

        # Compute average expert index weighted by scores
        # Shape: gate_top_k_idx (batch*seq, top_k), gate_score (batch*seq, top_k)
        avg_expert_idx = (gate_top_k_idx.float() * gate_score).sum(dim=-1).mean()

        # dummy loss
        self.set_loss(torch.zeros(1, requires_grad=True).to(inp.device))

        if return_all_scores:
            return gate_top_k_idx, gate_score, gate, avg_expert_idx
        return gate_top_k_idx, gate_score, avg_expert_idx

class FMoEViz(FMoE):
    def forward(self, moe_inp):
        r"""
        The FMoE module first computes gate output, and then conduct MoE forward
        according to the gate.  The score of the selected gate given by the
        expert is multiplied to the experts' output tensors as a weight.
        """

        moe_inp_batch_size = tree.flatten(
            tree.map_structure(lambda tensor: tensor.shape[0], moe_inp)
        )
        assert all(
            [batch_size == moe_inp_batch_size[0] for batch_size in moe_inp_batch_size]
        ), "MoE inputs must have the same batch size"

        if self.world_size > 1:

            def ensure_comm_func(tensor):
                ensure_comm(tensor, self.moe_group)

            tree.map_structure(ensure_comm_func, moe_inp)
        if self.slice_size > 1:

            def slice_func(tensor):
                return Slice.apply(
                    tensor, self.slice_rank, self.slice_size, self.slice_group
                )

            moe_inp = tree.map_structure(slice_func, moe_inp)

        gate_top_k_idx, gate_score = self.gate(moe_inp)

        if self.gate_hook is not None:
            self.gate_hook(gate_top_k_idx, gate_score, None)

        # delete masked tensors
        if self.mask is not None and self.mask_dict is not None:
            # TODO: to fix
            def delete_mask_func(tensor):
                # to: (BxL') x d_model
                tensor = tensor[mask == 0, :]
                return tensor

            mask = self.mask.view(-1)
            moe_inp = tree.map_structure(delete_mask_func, moe_inp)
            gate_top_k_idx = gate_top_k_idx[mask == 0, :]

        fwd = _fmoe_general_global_forward(
            moe_inp, gate_top_k_idx, self.expert_fn_single if fmoe_faster_schedule else self.expert_fn,
            self.num_expert, self.world_size,
            experts=self.experts
        )

        # recover deleted tensors
        if self.mask is not None and self.mask_dict is not None:

            def recover_func(tensor):
                # to: (BxL') x top_k x dim
                dim = tensor.shape[-1]
                tensor = tensor.view(-1, self.top_k, dim)
                # to: (BxL) x top_k x d_model
                x = torch.zeros(
                    mask.shape[0],
                    self.top_k,
                    dim,
                    device=tensor.device,
                    dtype=tensor.dtype,
                )
                # recover
                x[mask == 0] = tensor
                for k, v in self.mask_dict.items():
                    x[mask == k] = v
                return x

            moe_outp = tree.map_structure(recover_func, fwd)
        else:

            def view_func(tensor):
                dim = tensor.shape[-1]
                tensor = tensor.view(-1, self.top_k, dim)
                return tensor

            moe_outp = tree.map_structure(view_func, fwd)

        gate_score = gate_score.view(-1, 1, self.top_k)

        def bmm_func(tensor):
            dim = tensor.shape[-1]
            tensor = torch.bmm(gate_score, tensor).reshape(-1, dim)
            return tensor

        moe_outp = tree.map_structure(bmm_func, moe_outp)

        if self.slice_size > 1:

            def all_gather_func(tensor):
                return AllGather.apply(
                    tensor, self.slice_rank, self.slice_size, self.slice_group
                )

            moe_outp = tree.map_structure(all_gather_func, moe_outp)

        moe_outp_batch_size = tree.flatten(
            tree.map_structure(lambda tensor: tensor.shape[0], moe_outp)
        )
        assert all(
            [batch_size == moe_outp_batch_size[0] for batch_size in moe_outp_batch_size]
        ), "MoE outputs must have the same batch size"
        return moe_outp, gate_score, gate_top_k_idx

class FMoETransformerMLPViz(FMoEViz):
    r"""
    A complete MoE MLP module in a Transformer block.
    * `activation` is the activation function to be used in MLP in each expert.
    * `d_hidden` is the dimension of the MLP layer.
    """

    def __init__(
        self,
        num_expert=32,
        d_model=1024,
        d_hidden=4096,
        activation=torch.nn.GELU(),
        expert_dp_comm="none",
        expert_rank=0,
        **kwargs
    ):
        def one_expert(d_model):
            return _ExpertPrint(1, d_model, d_hidden, activation, rank=0)
        
        expert = one_expert
        super().__init__(num_expert=num_expert, d_model=d_model, expert=expert, **kwargs)
        self.mark_parallel_comm(expert_dp_comm)

    def forward(self, inp: torch.Tensor, ret={}):
        r"""
        This module wraps up the FMoE module with reshape, residual and layer
        normalization.
        """
        original_shape = inp.shape
        inp = inp.reshape(-1, self.d_model)
        output, gate_score, top_k_idx = super().forward(inp)
        ret['gate_score'] = gate_score.reshape(original_shape[0], -1, 2)
        ret['top_k_idx'] = top_k_idx.reshape(original_shape[0], -1, 2)
        return output.reshape(original_shape)

class FMoEHarmonic(FMoE):
    def __init__(
        self,
        num_expert=32,
        d_model=1024,
        world_size=1,
        mp_group=None,
        slice_group=None,
        moe_group=None,
        top_k=2,
        gate=NaiveGate,
        expert=None,
        gate_hook=None,
        mask=None,
        mask_dict=None,
        gate_bias=True,
        d_hidden=None,
        expert_list=None,  # NEW: pass pre-built experts directly
    ):
        super().__init__()
        self.num_expert = num_expert
        self.d_model = d_model
        self.world_size = world_size

        self.slice_group = slice_group
        if mp_group is not None:
            print("[Warning] mp_group is being deprecated")
            self.slice_group = mp_group
        if self.slice_group is None:
            self.slice_size = 1
            self.slice_rank = 0
        else:
            self.slice_size = self.slice_group.size()
            self.slice_rank = self.slice_group.rank()

        self.top_k = top_k
        
        # NEW: support pre-built heterogeneous expert list
        if expert_list is not None:
            self.experts = nn.ModuleList(expert_list)
            self.experts_fused = False
            self.num_expert = len(expert_list)
        elif type(expert) is list:
            self.experts = nn.ModuleList([e(d_model) for e in expert])
            self.experts_fused = False
            self.num_expert = num_expert = len(expert)
        elif expert is not None:
            self.experts = nn.ModuleList([expert(d_model) for _ in range(num_expert)])
            self.experts_fused = False
        else:
            self.experts_fused = True

        if issubclass(gate, NaiveGate):
            self.gate = gate(d_model, num_expert, world_size, top_k, gate_bias=gate_bias)
        else:
            self.gate = gate(d_model, num_expert, world_size, top_k)
        self.gate_hook = gate_hook
        self.mask = mask
        self.mask_dict = mask_dict
        self.moe_group = moe_group

    def expert_fn(self, inp, fwd_expert_count):
        r"""
        Optimized expert function that batches operations where possible.
        Avoids sequential expert calls by grouping experts with same hidden size.
        """
        if self.experts_fused:
            return self.experts(inp, fwd_expert_count)
        
        if isinstance(fwd_expert_count, torch.Tensor):
            fwd_expert_count_cpu = fwd_expert_count.cpu().numpy()
        
        outputs = []
        base_idx = 0
        
        for i in range(self.num_expert):
            batch_size = fwd_expert_count_cpu[i]
            if batch_size > 0:
                inp_slice = inp[base_idx : base_idx + batch_size]
                # Use fwd_expert_count[i:i+1] to keep tensor shape for FMoELinear
                expert_out = self.experts[i](inp_slice, fwd_expert_count[i:i+1])
                outputs.append(expert_out)
                base_idx += batch_size
        
        return torch.cat(outputs, dim=0) if outputs else inp[:0]
    def forward(self, moe_inp):
        r"""
        The FMoE module first computes gate output, and then conduct MoE forward
        according to the gate.  The score of the selected gate given by the
        expert is multiplied to the experts' output tensors as a weight.
        """
        inp_shape = moe_inp.shape
        if len(inp_shape) > 2: # done for class_token_moe
            inp_orig = moe_inp
            moe_inp = moe_inp[:, 0, :] if len(inp_shape) == 3 else moe_inp[:, 0, 0, :]

        moe_inp_batch_size = tree.flatten(
            tree.map_structure(lambda tensor: tensor.shape[0], moe_inp)
        )
        assert all(
            [batch_size == moe_inp_batch_size[0] for batch_size in moe_inp_batch_size]
        ), "MoE inputs must have the same batch size"

        if self.world_size > 1:

            def ensure_comm_func(tensor):
                ensure_comm(tensor, self.moe_group)

            tree.map_structure(ensure_comm_func, moe_inp)
        if self.slice_size > 1:

            def slice_func(tensor):
                return Slice.apply(
                    tensor, self.slice_rank, self.slice_size, self.slice_group
                )

            moe_inp = tree.map_structure(slice_func, moe_inp)

        gate_top_k_idx, gate_score, avg_expert_idx = self.gate(moe_inp)

        if len(inp_shape) > 2: # done for class_token_moe
            num_tokens = inp_shape[1] if len(inp_shape) == 3 else inp_shape[1] * inp_shape[1]
            gate_top_k_idx = gate_top_k_idx.repeat_interleave(num_tokens, 0)  # (batch_size, top_k)
            gate_score = gate_score.repeat_interleave(num_tokens, 0)  # (batch_size, top_k)
            moe_inp = inp_orig.reshape(-1, inp_shape[-1]) # use original input for expert forward

        if self.gate_hook is not None:
            self.gate_hook(gate_top_k_idx, gate_score, None)

        # delete masked tensors
        if self.mask is not None and self.mask_dict is not None:
            # TODO: to fix
            def delete_mask_func(tensor):
                # to: (BxL') x d_model
                tensor = tensor[mask == 0, :]
                return tensor

            mask = self.mask.view(-1)
            moe_inp = tree.map_structure(delete_mask_func, moe_inp)
            gate_top_k_idx = gate_top_k_idx[mask == 0, :]

        fwd = _fmoe_general_global_forward(
            moe_inp, gate_top_k_idx, self.expert_fn_single if fmoe_faster_schedule else self.expert_fn,
            self.num_expert, self.world_size,
            experts=self.experts
        )

        # recover deleted tensors
        if self.mask is not None and self.mask_dict is not None:

            def recover_func(tensor):
                # to: (BxL') x top_k x dim
                dim = tensor.shape[-1]
                tensor = tensor.view(-1, self.top_k, dim)
                # to: (BxL) x top_k x d_model
                x = torch.zeros(
                    mask.shape[0],
                    self.top_k,
                    dim,
                    device=tensor.device,
                    dtype=tensor.dtype,
                )
                # recover
                x[mask == 0] = tensor
                for k, v in self.mask_dict.items():
                    x[mask == k] = v
                return x

            moe_outp = tree.map_structure(recover_func, fwd)
        else:

            def view_func(tensor):
                dim = tensor.shape[-1]
                tensor = tensor.view(-1, self.top_k, dim)
                return tensor

            moe_outp = tree.map_structure(view_func, fwd)

        gate_score = gate_score.view(-1, 1, self.top_k)

        def bmm_func(tensor):
            dim = tensor.shape[-1]
            tensor = torch.bmm(gate_score, tensor).reshape(-1, dim)
            return tensor

        moe_outp = tree.map_structure(bmm_func, moe_outp)

        if self.slice_size > 1:

            def all_gather_func(tensor):
                return AllGather.apply(
                    tensor, self.slice_rank, self.slice_size, self.slice_group
                )

            moe_outp = tree.map_structure(all_gather_func, moe_outp)

        moe_outp_batch_size = tree.flatten(
            tree.map_structure(lambda tensor: tensor.shape[0], moe_outp)
        )
        assert all(
            [batch_size == moe_outp_batch_size[0] for batch_size in moe_outp_batch_size]
        ), "MoE outputs must have the same batch size"

        if viz: return moe_outp, avg_expert_idx, gate_score, gate_top_k_idx
        return moe_outp, avg_expert_idx


class _IdentityExpert(nn.Module):
    r"""
    Identity expert that passes input straight through.
    """
    def forward(self, inp, fwd_expert_count):
        return inp


class FMoETransformerMLPHarmonic(FMoEHarmonic):
    r"""
    Optimized heterogeneous MoE MLP with 8 experts of varying hidden sizes.
    """
    def __init__(
        self,
        num_expert=8,
        d_model=1024,
        d_hidden=4096,
        activation=torch.nn.GELU(),
        expert_dp_comm="none",
        expert_rank=0,
        top_k=2,
        class_token_moe=False,
        **kwargs
    ):
        # Build heterogeneous expert list
        ratios = [1.0/14, 1.0/12, 1.0/10, 1.0/8, 1.0/6, 1.0/4, 1.0/2]
        expert_list = [_IdentityExpert()]  # Expert 0: identity
        
        # Experts 1-7: variable hidden sizes
        for ratio in ratios:
            hidden = int(d_hidden * ratio)
            expert_list.append(_Expert(1, d_model, hidden, activation, rank=expert_rank))
        
        # Pass pre-built experts directly for efficiency
        super().__init__(
            num_expert=num_expert,
            d_model=d_model,
            expert_list=expert_list,
            top_k=top_k,
            **kwargs
        )
        self.mark_parallel_comm(expert_dp_comm)
        self.class_token_moe = class_token_moe

    def forward(self, inp: torch.Tensor, ret={}):
        original_shape = inp.shape
        if not self.class_token_moe: inp = inp.reshape(-1, self.d_model)
        if viz: 
            output, avg_expert_idx, gate_score, top_k_idx = super().forward(inp)
            ret["gate_score"] = gate_score.reshape(original_shape[0], -1, 2)
            ret["top_k_idx"] = top_k_idx.reshape(original_shape[0], -1, 2)
        else: output, avg_expert_idx = super().forward(inp)
        ret["avg_expert_idx"] = avg_expert_idx
        return output.reshape(original_shape)


def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True
    
def load_config(config_path):
    with open(config_path, 'r') as f:
        opts = yaml.safe_load(f)
    opts = Box(opts)
    return opts

def get_th(opts, model):
    th = round(model.hyper_encoders[0].group_gen.th.item(), 4)
    return th    


if viz:
    if even_harmonic: moe_transformer_mlp, moe_gate = FMoETransformerMLPHarmonic, GSoftmaxHarmonicGate
    else: moe_transformer_mlp, moe_gate = FMoETransformerMLPViz, GSoftmaxGate

    if class_token: 
        moe_transformer_mlp_noncls = FMoETransformerMLPViz # makes it so that first 2 layers use equal div1 experts and aren't included in bias loss
        moe_gate_noncls = GSoftmaxGate
    else: moe_transformer_mlp_noncls, moe_gate_noncls = None, None
else:
    if even_harmonic: moe_transformer_mlp, moe_gate = FMoETransformerMLPHarmonic, GSoftmaxHarmonicGate
    else: moe_transformer_mlp, moe_gate = FMoETransformerMLP, GSoftmaxGate

    if even_harmonic and class_token: 
        moe_transformer_mlp_noncls = FMoETransformerMLP # makes it so that first 2 layers use equal div1 experts and aren't included in bias loss
        moe_gate_noncls = GSoftmaxGate
    else: moe_transformer_mlp_noncls, moe_gate_noncls = None, None

