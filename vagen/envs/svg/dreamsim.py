import torch
from PIL import Image
import os
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import List, Any

# --- peft / torchao compatibility shim -------------------------------------
# peft 0.19.1's is_torchao_available() hard-raises ImportError when torchao<0.16,
# even though its own dispatch_torchao() is designed to *skip* the torchao path
# (return None) when torchao is unavailable. This breaks dreamsim, which injects
# LoRA via get_peft_model -> _create_new_module -> dispatch_torchao.
#
# The run env pins torchao==0.9.0 because sglang 0.5.2 imports
# `float8_dynamic_activation_float8_weight` from torchao.quantization at model
# init (that symbol was renamed/removed in torchao>=0.16). So we cannot upgrade
# torchao without breaking sglang, and we cannot downgrade peft without risking
# other compat. Instead we make is_torchao_available() return False for the
# <0.16 case (matching peft's own intent), so dispatch_torchao() falls through
# to the standard LoRA path. sglang uses torchao directly (not via this function)
# and is unaffected.
try:
    import peft.import_utils as _peft_import_utils
    import packaging.version

    def _safe_is_torchao_available():
        try:
            import importlib.util
            if importlib.util.find_spec("torchao") is None:
                return False
            import importlib.metadata as _meta
            _v = packaging.version.parse(_meta.version("torchao"))
            if _v < packaging.version.parse("0.16.0"):
                # Old torchao (e.g. 0.9.0 required by sglang): do not use the
                # torchao LoRA dispatcher; fall back to standard LoRA.
                return False
            return True
        except Exception:
            return False

    # Patch both the canonical definition and the name already bound in the
    # lora.torchao module (dispatch_torchao references the latter).
    _peft_import_utils.is_torchao_available = _safe_is_torchao_available
    try:
        import peft.tuners.lora.torchao as _peft_lora_torchao
        _peft_lora_torchao.is_torchao_available = _safe_is_torchao_available
    except Exception:
        pass
except Exception:
    pass
# --- end shim ---------------------------------------------------------------

from dreamsim import dreamsim

class DreamSimScoreCalculator:
    """
    A wrapper class for DreamSim model to calculate similarity scores between images.
    """

    def __init__(self, pretrained=True, cache_dir="~/.cache", device=None):
        """
        Initialize DreamSim model.
        """
        cache_dir = os.path.expanduser(cache_dir)

        import torch
        # Verify device availability
        if device is None:
            self.device = "cpu"
        elif isinstance(device, str) and device.startswith("cuda:") and not torch.cuda.is_available():
            self.device = "cpu"
        else:
            self.device = device

        # Prevent torch.hub from network access
        torch.hub.set_dir(os.path.join(cache_dir, ""))

        # Load model and preprocessor
        self.model, self.preprocess = dreamsim(pretrained=pretrained, cache_dir=cache_dir, device=self.device)

        # DreamSim's feature extractors (dino ViT etc.) use forward hooks to grab
        # intermediate activations into shared buffers. Concurrent forward calls
        # (svg_env runs scoring in a ThreadPoolExecutor via asyncio.to_thread, and
        # all envs share one process-level singleton) race on those hooks and mix
        # activations between threads, producing
        # "Tensors must have same number of dimensions: got 2 and 1" from the
        # internal torch.cat of differently-shaped squeezed features. Serialize the
        # forward pass with a per-instance lock. The models are tiny, so this is
        # cheap and keeps GPU memory O(1) via the singleton in svg_env.
        import threading
        self._fwd_lock = threading.Lock()

    def calculate_similarity_score(self, gt_im, gen_im):
        """
        Calculate similarity score between ground truth and generated images.
        """
        # Preprocess images
        img1 = self.preprocess(gt_im)
        img2 = self.preprocess(gen_im)

        # Move to device if necessary
        img1 = img1.to(self.device)
        img2 = img2.to(self.device)

        # Calculate distance (lower is better). Serialize the forward pass across
        # threads (see _fwd_lock docstring) to avoid the dino-extractor hook race.
        with self._fwd_lock, torch.no_grad():
            distance = self.model(img1, img2).item()

        # Convert distance to similarity score (1 - normalized distance)
        similarity = 1.0 - min(1.0, max(0.0, distance))

        return similarity

    def calculate_batch_scores(self, gt_images: List[Any], gen_images: List[Any]) -> List[float]:
        """
        Calculate similarity scores for multiple image pairs.
        Since DreamSim doesn't natively support batch comparison, we process each pair individually.
        """
        if not gt_images or not gen_images:
            return []

        batch_size = len(gt_images)

        gt_processed = [self.preprocess(img).to(self.device) for img in gt_images]
        gen_processed = [self.preprocess(img).to(self.device) for img in gen_images]

        scores = []
        for i in range(batch_size):
            with self._fwd_lock, torch.no_grad():
                distance = self.model(gt_processed[i], gen_processed[i]).item()
            similarity = 1.0 - min(1.0, max(0.0, distance))
            scores.append(similarity)

        return scores
