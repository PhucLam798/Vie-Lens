"""Selective Hugging Face forward-hook recording for residual streams."""

from __future__ import annotations

from dataclasses import dataclass
from operator import attrgetter
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch

from .models import LoadedModel


@dataclass
class Recording:
    """Selected residuals, optional block components, and model logits."""

    resid: torch.Tensor
    logits: torch.Tensor
    attn: torch.Tensor | None = None
    mlp: torch.Tensor | None = None


def record(
    loaded: LoadedModel,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    *,
    positions: torch.Tensor,
    components: bool = False,
    keep_on: str | torch.device = "cpu",
) -> Recording:
    """Run one forward pass and retain only requested sequence positions."""
    import torch

    if positions.ndim != 2 or positions.shape[0] != input_ids.shape[0]:
        raise ValueError("positions must have shape [batch, selected_positions]")
    if positions.dtype != torch.long:
        raise ValueError("positions must be a LongTensor")
    if input_ids.ndim != 2 or attention_mask.shape != input_ids.shape:
        raise ValueError("input_ids and attention_mask must have matching [batch, sequence] shapes")

    handles: list[Any] = []
    resid: list[torch.Tensor | None] = [None] * (loaded.n_layers + 1)
    attn: list[torch.Tensor | None] = [None] * loaded.n_layers
    mlp: list[torch.Tensor | None] = [None] * loaded.n_layers
    output_logits: torch.Tensor | None = None

    def select(value: Any) -> torch.Tensor:
        """Gather requested positions from a full hidden-state tensor immediately."""
        if not isinstance(value, torch.Tensor) or value.ndim != 3:
            raise TypeError("hooked activation must be a [batch, sequence, hidden] tensor")
        index = positions.to(device=value.device).unsqueeze(-1).expand(-1, -1, value.shape[-1])
        return value.gather(1, index).detach().to(device=keep_on)

    def first_tensor(value: Any) -> torch.Tensor:
        """Return the first tensor of tuple-returning modules or the tensor itself."""
        selected = value[0] if isinstance(value, tuple) else value
        if not isinstance(selected, torch.Tensor):
            raise TypeError("hooked module output is not a tensor")
        return selected

    def layer_input_hook(
        _module: Any, args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> None:
        """Capture the text decoder input as residual layer zero."""
        if args:
            value = args[0]
        else:
            value = kwargs.get("hidden_states")
        if value is None:
            raise TypeError("first decoder layer received no hidden_states input")
        resid[0] = select(value)

    try:
        handles.append(loaded.layers[0].register_forward_pre_hook(layer_input_hook, with_kwargs=True))
        for layer_index, layer in enumerate(loaded.layers):
            handles.append(
                layer.register_forward_hook(
                    lambda _module, _args, output, i=layer_index: resid.__setitem__(i + 1, select(first_tensor(output)))
                )
            )
            if components:
                attn_module = attrgetter(loaded.module_paths["attn_out"])(layer)
                mlp_module = attrgetter(loaded.module_paths["mlp_out"])(layer)
                handles.append(
                    attn_module.register_forward_hook(
                        lambda _module, _args, output, i=layer_index: attn.__setitem__(i, select(first_tensor(output)))
                    )
                )
                handles.append(
                    mlp_module.register_forward_hook(
                        lambda _module, _args, output, i=layer_index: mlp.__setitem__(i, select(first_tensor(output)))
                    )
                )
        with torch.inference_mode():
            model_output = loaded.model(input_ids=input_ids, attention_mask=attention_mask)
        output_logits = getattr(model_output, "logits", None)
        if output_logits is None:
            if isinstance(model_output, (tuple, list)) and model_output:
                output_logits = model_output[0]
            else:
                raise TypeError("model output does not contain logits")
        logits_index = positions.to(device=output_logits.device).unsqueeze(-1).expand(
            -1, -1, output_logits.shape[-1]
        )
        selected_logits = output_logits.gather(1, logits_index).detach().to(device=keep_on)
        if any(value is None for value in resid):
            raise RuntimeError("one or more decoder residual hooks did not run")
        if components and (any(value is None for value in attn) or any(value is None for value in mlp)):
            raise RuntimeError("one or more component hooks did not run")
        stacked_resid = torch.stack([value for value in resid if value is not None], dim=0)
        stacked_attn = (
            torch.stack([value for value in attn if value is not None], dim=0) if components else None
        )
        stacked_mlp = torch.stack([value for value in mlp if value is not None], dim=0) if components else None
        return Recording(resid=stacked_resid, logits=selected_logits, attn=stacked_attn, mlp=stacked_mlp)
    finally:
        for handle in handles:
            handle.remove()
