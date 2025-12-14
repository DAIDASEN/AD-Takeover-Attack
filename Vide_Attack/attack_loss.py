import torch
from typing import Tuple

def verbose_losses_from_generate_output(
    outputs,
    eos_token_id: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Returns:
        loss_eos, loss_uncertainty, loss_diversity (each scalar tensor)
    """
    # outputs.scores is a tuple of len T, each is [batch, vocab_size]
    # Stack to [batch, T, vocab_size]
    scores = torch.stack(outputs.scores, dim=1)
    
    # 1. Delayed-EOS loss
    # probs = scores.log_softmax(dim=-1).exp()
    probs = torch.nn.functional.softmax(scores, dim=-1)
    
    # eos_probs = probs[..., eos_token_id] -> [batch, T]
    eos_probs = probs[..., eos_token_id]
    
    # loss_eos = eos_probs.mean() (minimize)
    loss_eos = eos_probs.mean()
    
    # 2. Uncertainty loss (maximize entropy)
    # H = -(probs * (probs.clamp_min(1e-8).log())).sum(dim=-1) -> [batch, T]
    log_probs = torch.nn.functional.log_softmax(scores, dim=-1)
    H = -(probs * log_probs).sum(dim=-1)
    
    # loss_uncertainty = -H.mean() (minimize; pushes entropy up)
    loss_uncertainty = -H.mean()
    
    # 3. Token diversity loss (maximize nuclear norm)
    # outputs.hidden_states is a tuple of len T (one per step)
    # Each element is a tuple of len num_layers (one per layer)
    # We want the last layer for each step.
    # outputs.hidden_states[t][-1] -> [batch, 1, hidden_dim] or [batch, hidden_dim] depending on model
    
    # Let's inspect the structure. Usually for causal LM generation:
    # hidden_states[0] contains the hidden states for the input prompt + first generated token?
    # No, usually generate returns hidden_states for the generated tokens only if return_dict_in_generate=True
    # and output_hidden_states=True.
    # It returns a tuple of tuples.
    # The outer tuple is over generated steps.
    # The inner tuple is over layers.
    
    hidden_states_list = []
    for step_hidden_states in outputs.hidden_states:
        # Get last layer
        last_layer = step_hidden_states[-1] # [batch, 1, hidden_dim] or [batch, hidden_dim]
        hidden_states_list.append(last_layer)
        
    # Stack to [batch, seq_len, hidden_dim]
    # Note: last_layer might be [batch, hidden_dim] or [batch, 1, hidden_dim]
    if hidden_states_list[0].dim() == 2:
        hidden_states = torch.stack(hidden_states_list, dim=1)
    else:
        hidden_states = torch.cat(hidden_states_list, dim=1)
        
    # Assume batch_size = 1 for this attack
    # Shape [1, seq_len, hidden_dim]
    # Flatten to H: [seq_len, hidden_dim]
    H_matrix = hidden_states.squeeze(0)
    
    # SVD
    # U, S, Vh = torch.linalg.svd(H, full_matrices=False)
    # nuclear_norm = S.sum()
    # loss_diversity = -nuclear_norm
    
    try:
        _, S, _ = torch.linalg.svd(H_matrix, full_matrices=False)
        nuclear_norm = S.sum()
        loss_diversity = -nuclear_norm
    except RuntimeError:
        # Fallback if SVD fails (e.g. NaNs or instability)
        loss_diversity = torch.tensor(0.0, device=scores.device, requires_grad=True)

    return loss_eos, loss_uncertainty, loss_diversity

def total_verbose_loss(loss_eos, loss_uncertainty, loss_diversity,
                       w_eos=1.0, w_uncertainty=1.0, w_diversity=1.0):
    return w_eos * loss_eos + w_uncertainty * loss_uncertainty + w_diversity * loss_diversity
