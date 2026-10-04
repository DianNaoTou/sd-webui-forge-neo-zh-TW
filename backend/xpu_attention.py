"""Bound temporary attention matrices on the Intel XPU math backend."""
import logging
import math

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

logger = logging.getLogger("attention")
_reported_slicing = False


def math_sdpa(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False,
              *, scale=None, enable_gqa=False):
    """Slice inference queries while retaining the complete key/value sequence.

    A query row attends independently to every key. Slicing that dimension
    therefore preserves non-causal attention, including broadcast masks and GQA.
    Keep dropout, causal and unusual-rank calls on the original implementation.
    The budget estimates FP32 score/softmax temporaries, not total GPU usage.
    """
    extra = {"enable_gqa": enable_gqa}
    if scale is not None:
        extra["scale"] = scale
    with sdpa_kernel(SDPBackend.MATH):
        if q.ndim != 4 or k.ndim != 4 or v.ndim != 4 or is_causal or dropout_p:
            return torch.nn.functional.scaled_dot_product_attention(
                q, k, v, attn_mask=attn_mask, dropout_p=dropout_p,
                is_causal=is_causal, **extra)

        batch_shape = torch.broadcast_shapes(q.shape[:-3], k.shape[:-3], v.shape[:-3])
        query_length, key_length = q.shape[-2], k.shape[-2]
        # Allow 128 MiB for four FP32 score-sized intermediates and at most
        # 256 query rows per call. The lower bound also handles very long keys.
        rows_bytes = math.prod(batch_shape) * q.shape[-3] * key_length * 4 * 4
        chunk = max(1, min(256, (128 * 1024 * 1024) // max(1, rows_bytes)))
        if query_length <= chunk:
            return torch.nn.functional.scaled_dot_product_attention(
                q, k, v, attn_mask=attn_mask, dropout_p=0.0,
                is_causal=False, **extra)

        global _reported_slicing
        if not _reported_slicing:
            logger.info("XPU Math SDPA query slicing enabled (chunk=%s, q=%s, k=%s)",
                        chunk, query_length, key_length)
            _reported_slicing = True

        out = torch.empty((*batch_shape, q.shape[-3], query_length, v.shape[-1]),
                          device=q.device, dtype=q.dtype)
        for start in range(0, query_length, chunk):
            end = min(start + chunk, query_length)
            mask = attn_mask
            if mask is not None and mask.ndim >= 2 and mask.shape[-2] != 1:
                mask = mask[..., start:end, :]
            piece = torch.nn.functional.scaled_dot_product_attention(
                q[..., start:end, :], k, v, attn_mask=mask,
                dropout_p=0.0, is_causal=False, **extra)
            out[..., start:end, :] = piece
            del piece
        return out
