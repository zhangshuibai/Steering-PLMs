"""
Custom steering forwards for alpha scaling experiments.

Variants:
- steering_forward_all_layer_with_glp: steer at ALL layers, GLP project only at glp_layer.
  (existing steering_forward_with_glp only steers at glp_layer.)
"""

import torch


def steering_forward_all_layer_with_glp(
    self, tokens, repr_layers=[], need_head_weights=False,
    return_contacts=False, steering_vectors=None,
    glp_project_fn=None, glp_layer=16
):
    """All-layer steering + L17-GLP projection.
    - Inject steering_vectors[layer_idx] after EVERY layer (like standard all-layer steering)
    - Apply GLP projection only after self.layers[glp_layer]
    """
    if return_contacts:
        need_head_weights = True

    assert tokens.ndim == 2
    padding_mask = tokens.eq(self.padding_idx)

    x = self.embed_scale * self.embed_tokens(tokens)
    if padding_mask is not None:
        x = x * (1 - padding_mask.unsqueeze(-1).type_as(x))

    repr_layers = set(repr_layers)
    hidden_representations = {}
    if 0 in repr_layers:
        hidden_representations[0] = x
    if need_head_weights:
        attn_weights = []

    x = x.transpose(0, 1)
    if not padding_mask.any():
        padding_mask = None

    for layer_idx, layer in enumerate(self.layers):
        x, attn = layer(x, self_attn_padding_mask=padding_mask, need_head_weights=need_head_weights)

        # Steering at every layer (with norm-preserving rescale)
        if steering_vectors is not None:
            add_x = steering_vectors[layer_idx]
            new_x = x + add_x
            new_x_norm = torch.norm(new_x, p=2, dim=-1, keepdim=True).detach()
            x_norm = torch.norm(x, p=2, dim=-1, keepdim=True).detach()
            x = new_x * (x_norm / new_x_norm)

        # GLP projection ONLY at glp_layer (after steering)
        if glp_project_fn is not None and layer_idx == glp_layer:
            x = glp_project_fn(x)

        if (layer_idx + 1) in repr_layers:
            hidden_representations[layer_idx + 1] = x.transpose(0, 1)
        if need_head_weights:
            attn_weights.append(attn.transpose(1, 0))

    x = self.emb_layer_norm_after(x)
    x = x.transpose(0, 1)
    if (layer_idx + 1) in repr_layers:
        hidden_representations[layer_idx + 1] = x
    x = self.lm_head(x)

    result = {"logits": x, "representations": hidden_representations}
    if need_head_weights:
        attentions = torch.stack(attn_weights, 1)
        if padding_mask is not None:
            attention_mask = 1 - padding_mask.type_as(attentions)
            attention_mask = attention_mask.unsqueeze(1) * attention_mask.unsqueeze(2)
            attentions = attentions * attention_mask[:, None, None, :, :]
        result["attentions"] = attentions
        if return_contacts:
            contacts = self.contact_head(tokens, attentions)
            result["contacts"] = contacts
    return result
