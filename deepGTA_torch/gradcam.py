"""
gradcam.py
==========
FIX #22: Grad-CAM visualisation for DeepSKAN.

Grad-CAM highlights which regions of the TA map (time × wavelength) most
influenced the CNN's classification decision.  This helps researchers:
  - Verify the model is looking at physically meaningful features
    (e.g., rise/decay in the right time window, correct spectral band)
  - Identify failure modes when confidence is low
  - Build trust in the prediction before committing to GA/TA fitting

Reference: Selvaraju et al. "Grad-CAM: Visual Explanations from Deep Networks
via Gradient-based Localization." ICCV 2017.

Usage
-----
    from deepGTA_torch.gradcam import GradCAM, plot_gradcam

    gcam = GradCAM(model, target_layer=model.res7)
    cam  = gcam.compute(X_cnn, class_id=predicted_class)
    fig  = plot_gradcam(X_cnn[0, 0], cam, title="Grad-CAM")
    fig.savefig("results/gradcam.png", dpi=150)
"""

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


class GradCAM:
    """
    Gradient-weighted Class Activation Mapping for DeepSKAN.

    Hooks onto any Conv2d layer and computes the gradient of the target
    class logit with respect to the feature maps at that layer.
    The resulting heatmap has the same aspect ratio as the feature map
    and is up-sampled to match the original input size.

    Args:
        model:        A trained DeepSKAN instance.
        target_layer: The nn.Module to hook (default: model.res7 — the last
                      convolutional block before global pooling).
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module = None):
        self.model        = model
        self.target_layer = target_layer or self._find_last_conv(model)

        self._gradients: Optional[torch.Tensor] = None
        self._activations: Optional[torch.Tensor] = None

        # Register forward and backward hooks
        self._fwd_hook = self.target_layer.register_forward_hook(self._save_activation)
        self._bwd_hook = self.target_layer.register_full_backward_hook(self._save_gradient)

    # ── Hook callbacks ────────────────────────────────────────────────────────

    def _save_activation(self, module, input, output):
        self._activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self._gradients = grad_output[0].detach()

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _find_last_conv(model: nn.Module) -> nn.Module:
        """Walk the model and return the last Conv2d layer found."""
        last = None
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                last = m
        if last is None:
            raise ValueError("No Conv2d layers found in model.")
        return last

    # ── Main API ──────────────────────────────────────────────────────────────

    def compute(
        self,
        X: np.ndarray,
        class_id: int,
        input_size: tuple = (256, 64),
    ) -> np.ndarray:
        """
        Compute Grad-CAM heatmap for a given input and class.

        Args:
            X:          Input array — shape (1, 256, 64) or (B, 1, 256, 64).
            class_id:   Target class index (e.g. the CNN top-1 prediction).
            input_size: Original (H, W) of the input — used for up-sampling.

        Returns:
            cam: float32 numpy array of shape (H, W), values in [0, 1].
                 Larger values = more important for classifying as class_id.
        """
        self.model.eval()

        X_np = np.array(X, dtype=np.float32)
        if X_np.ndim == 3:
            X_np = X_np[np.newaxis]   # → (1, 1, H, W)

        device    = next(self.model.parameters()).device
        X_tensor  = torch.tensor(X_np, requires_grad=False).to(device)

        # Forward pass — need gradients, so can't use torch.no_grad()
        self.model.zero_grad()
        logits = self.model(X_tensor)        # (1, num_classes)

        # Backward pass for the target class
        score = logits[0, class_id]
        score.backward()

        # Grad-CAM: global average pool gradients over spatial dims
        # grads: (1, C, h, w)  activations: (1, C, h, w)
        grads       = self._gradients          # (1, C, h, w)
        activations = self._activations        # (1, C, h, w)

        weights = grads.mean(dim=(2, 3), keepdim=True)    # (1, C, 1, 1)
        cam_raw = (weights * activations).sum(dim=1)       # (1, h, w)
        cam_raw = torch.relu(cam_raw)[0].cpu().numpy()     # (h, w)

        # Up-sample to input resolution using bilinear interpolation
        from scipy.ndimage import zoom
        if cam_raw.shape != input_size:
            scale = (input_size[0] / cam_raw.shape[0],
                     input_size[1] / cam_raw.shape[1])
            cam_up = zoom(cam_raw, scale, order=1)
        else:
            cam_up = cam_raw

        # Normalise to [0, 1]
        cam_min, cam_max = cam_up.min(), cam_up.max()
        if cam_max > cam_min:
            cam_up = (cam_up - cam_min) / (cam_max - cam_min)
        else:
            cam_up = np.zeros_like(cam_up)

        return cam_up.astype(np.float32)

    def remove_hooks(self):
        """Call this when done to avoid memory leaks from persistent hooks."""
        self._fwd_hook.remove()
        self._bwd_hook.remove()


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_gradcam(
    X_input: np.ndarray,
    cam: np.ndarray,
    title: str = 'Grad-CAM',
    t: np.ndarray = None,
    wl: np.ndarray = None,
    alpha: float = 0.45,
) -> plt.Figure:
    """
    Overlay the Grad-CAM heatmap on top of the TA input image.

    Args:
        X_input:  2-D float array (H, W) — the raw TA map (time × wavelength).
        cam:      2-D float array (H, W) — Grad-CAM output from GradCAM.compute().
        title:    Figure title.
        t:        Optional time axis (ps) for proper y-axis labels.
        wl:       Optional wavelength axis (nm) for proper x-axis labels.
        alpha:    Transparency of the CAM overlay (0 = invisible, 1 = opaque).

    Returns:
        matplotlib Figure with two subplots: raw TA and TA + CAM overlay.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    x_vals = wl if wl is not None else np.arange(X_input.shape[1])
    y_vals = t  if t  is not None else np.arange(X_input.shape[0])

    # Common colour scale centred on zero for TA
    v_abs  = max(np.nanmax(np.abs(X_input)), 1e-6)
    levels = np.linspace(-v_abs, v_abs, 64)

    # ── Left: raw TA ──────────────────────────────────────────────────────
    ax = axes[0]
    c  = ax.contourf(x_vals, y_vals, X_input, levels=levels, cmap='RdBu_r')
    fig.colorbar(c, ax=ax, label='ΔA (norm.)')
    ax.set_title('Input TA Map', fontsize=12)
    ax.set_xlabel('Wavelength (nm)' if wl is not None else 'Wavelength (px)')
    ax.set_ylabel('Time (ps)'       if t  is not None else 'Time (px)')
    if t is not None and np.all(t > 0):
        ax.set_yscale('log')

    # ── Right: TA + Grad-CAM overlay ─────────────────────────────────────
    ax = axes[1]
    ax.contourf(x_vals, y_vals, X_input, levels=levels, cmap='RdBu_r')

    # Jet heatmap for the CAM; only show positive values
    cam_rgba = plt.cm.jet(cam)
    cam_rgba[..., 3] = cam * alpha    # alpha proportional to importance

    ax.imshow(
        cam_rgba,
        aspect='auto',
        origin='lower',
        extent=[x_vals[0], x_vals[-1], y_vals[0], y_vals[-1]],
        interpolation='bilinear',
    )

    # Colourbar proxy for the CAM scale
    sm = plt.cm.ScalarMappable(cmap='jet', norm=mcolors.Normalize(0, 1))
    sm.set_array([])
    fig.colorbar(sm, ax=ax, label='Grad-CAM importance')

    ax.set_title(title, fontsize=12)
    ax.set_xlabel('Wavelength (nm)' if wl is not None else 'Wavelength (px)')
    ax.set_ylabel('Time (ps)'       if t  is not None else 'Time (px)')
    if t is not None and np.all(t > 0):
        ax.set_yscale('log')

    fig.suptitle('Grad-CAM: CNN Decision Explanation', fontsize=13, y=1.01)
    fig.tight_layout()
    return fig
