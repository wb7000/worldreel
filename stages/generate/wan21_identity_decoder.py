"""
WorldReel - Character-consistent .wreel generation pipeline.

This module provides the core generation pipeline that takes latent vectors
and produces .wreel-compatible video frames with consistent character identity.

Key components:
- FaceEmbeddingProjector: Encodes a portrait image into a face identity embedding.
- WanDecoder3dWithIdentity: Wraps Wan2.1 VAE decoder to inject face identity
  into every upsampling layer, ensuring consistent character appearance across
  all generated frames and scenes.
- create_identity_decoder: Factory function to build the decoder.

Usage:
    from stages.generate.wan21_identity_decoder import (
        WanDecoder3dWithIdentity,
        FaceEmbeddingProjector,
        create_identity_decoder,
    )

    # Load base decoder
    decoder = WanDecoder3d(...)  # from diffusers

    # Wrap with identity injection
    decoder_with_id = WanDecoder3dWithIdentity(decoder, face_embedding_dim=512)

    # Generate with face identity
    frames = decoder_with_id(latents, face_embedding=face_emb)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from diffusers.models.autoencoders.autoencoder_kl_wan import WanDecoder3d


class FaceEmbeddingProjector(nn.Module):
    """Projects a face embedding (from YuNet/SFace or similar) to decoder feature dimensions."""

    def __init__(self, face_embedding_dim: int = 512, target_dim: int = 512):
        super().__init__()
        self.projector = nn.Sequential(
            nn.Linear(face_embedding_dim, target_dim * 2),
            nn.ReLU(inplace=True),
            nn.Linear(target_dim * 2, target_dim),
            nn.Tanh(),
        )

    def forward(self, face_embedding: torch.Tensor) -> torch.Tensor:
        """Project face embedding to decoder feature dimension.

        Args:
            face_embedding: [B, face_embedding_dim] face embedding vector
        Returns:
            [B, target_dim] projected face embedding
        """
        return self.projector(face_embedding)  # type: ignore[no-any-return]


class WanDecoder3dWithIdentity(nn.Module):
    """Wraps WanDecoder3d with face identity injection.

    Injects face embeddings into decoder feature maps at every upsampling layer,
    ensuring consistent face identity across all generated frames and scenes.

    Args:
        base_decoder: Pre-trained WanDecoder3d instance
        face_embedding_dim: Dimension of input face embeddings (default 512 for SFace)
    """

    def __init__(self, base_decoder: WanDecoder3d, face_embedding_dim: int = 512):
        super().__init__()
        self.base_decoder = base_decoder
        self.face_embedding_dim = face_embedding_dim

        # Copy attributes from base decoder
        self.dim = base_decoder.dim
        self.z_dim = base_decoder.z_dim
        self.dim_mult = base_decoder.dim_mult
        self.num_res_blocks = base_decoder.num_res_blocks
        self.attn_scales = base_decoder.attn_scales
        self.temperal_upsample = base_decoder.temperal_upsample
        self.nonlinearity = base_decoder.nonlinearity

        # Reuse base decoder sub-modules
        self.conv_in = base_decoder.conv_in
        self.mid_block = base_decoder.mid_block
        self.up_blocks = base_decoder.up_blocks
        self.norm_out = base_decoder.norm_out
        self.conv_out = base_decoder.conv_out
        self.gradient_checkpointing = base_decoder.gradient_checkpointing

        # Face embedding projector
        self.face_projector = FaceEmbeddingProjector(
            face_embedding_dim=face_embedding_dim,
            target_dim=self.dim,
        )

        # Face injection layers - one per upsampling block
        # Inject face embedding into each up_block's input features.
        # We create projectors for all possible channel dimensions and
        # dynamically select the right one at forward time.
        dims = [self.dim * u for u in [self.dim_mult[-1], *self.dim_mult[::-1]]]
        self.face_injectors = nn.ModuleDict()
        for out_dim in dims[:-1]:  # Inject into input of each up_block
            self.face_injectors[str(out_dim)] = nn.Linear(face_embedding_dim, out_dim)

    def forward(
        self,
        x: torch.Tensor,
        face_embedding: torch.Tensor | None = None,
        feat_cache: dict[Any, torch.Tensor] | None = None,
        feat_idx: list[int] | None = None,
        first_chunk: bool = False,
    ) -> torch.Tensor:
        """
        Decode latents with face identity injection.

        Args:
            x: Latent tensor [B, C, T, H, W]
            face_embedding: Face embedding [B, face_embedding_dim] or None
            feat_cache: Feature cache for causal conv (WorldReel compatibility)
            feat_idx: Feature index for cache management
            first_chunk: Whether this is the first chunk

        Returns:
            Decoded video frames [B, C, T, H, W]
        """
        # --- Step 1: Initial conv_in (same as real decoder) ---
        if feat_cache is not None and feat_idx is not None:
            idx = feat_idx[0]
            key = f"enc_{idx}"
            cache_x = x[:, :, -2:, :, :].clone()  # CACHE_T = 2
            if cache_x.shape[2] < 2 and feat_cache[key] is not None:
                cache_x = torch.cat(
                    [feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device), cache_x],
                    dim=2,
                )
            x = self.conv_in(x, feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            x = self.conv_in(x)

        # --- Step 2: Middle block (same as real decoder) ====================
        x = self.mid_block(x, feat_cache=feat_cache, feat_idx=feat_idx)

        # === Step 3: Upsample blocks with face injection ====================
        # Inject face embedding into each up_block's input features.
        for up_block in self.up_blocks:
            if face_embedding is not None:
                # Dynamically project face_embedding to match x's channel dimension
                c = x.shape[1]
                key = str(c)
                if key not in self.face_injectors:
                    self.face_injectors[key] = nn.Linear(self.face_embedding_dim, c).to(x.device)
                face_feat = self.face_injectors[key](face_embedding)  # [B, c]
                b, c = face_feat.shape
                face_feat = face_feat.view(b, c, 1, 1, 1)  # [B, C, 1, 1, 1]
                x = x + face_feat.expand_as(x)  # Broadcast to [B, C, T, H, W]
            x = up_block(x, feat_cache=feat_cache, feat_idx=feat_idx)

        # === Step 4: Output norm and conv (same as base decoder) ============
        x = self.norm_out(x)
        x = self.nonlinearity(x)
        x = self.conv_out(x)

        return x

    def decode_with_identity(
        self,
        latents: torch.Tensor,
        face_embedding: torch.Tensor | None = None,
        return_dict: bool = True,
    ) -> torch.Tensor:
        """
        Convenience method that wraps the full decode pipeline.

        Args:
            latents: Latent tensor [B, C, T, H, W]
            face_embedding: Face embedding [B, face_embedding_dim]
            return_dict: Whether to return dict or plain tensor

        Returns:
            Decoded video frames [B, C, T, H, W]
        """
        return self.forward(latents, face_embedding=face_embedding)


class FaceEmbeddingEncoder(nn.Module):
    """
    Encodes a face image into an identity embedding using a pretrained face
    recognition model (SFace or similar).

    This is used to create the face_embedding input for the decoder.
    """

    def __init__(self, embedding_dim: int = 512):
        super().__init__()
        self.embedding_dim = embedding_dim
        # Simple CNN encoder for face images
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=2, padding=1),  # [B, 64, H/2, W/2]
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),  # [B, 128, H/4, W/4]
            nn.ReLU(inplace=True),
            torch.nn.Conv2d(128, 256, 3, stride=2, padding=1),  # [B, 256, H/8, W/8]
            nn.ReLU(inplace=True),
            torch.nn.AdaptiveAvgPool2d((1, 1)),  # [B, 256, 1, 1]
            torch.nn.Flatten(),  # [B, 256]
            torch.nn.Linear(256, embedding_dim),  # [B, embedding_dim]
            nn.Tanh(),  # Bound embeddings
        )

    def forward(self, face_image: torch.Tensor) -> torch.Tensor:
        """Encode a face image into an identity embedding.

        Args:
            face_image: [B, 3, H, W] face image in [0, 1] range
        Returns:
            [B, embedding_dim] face identity embedding
        """
        return self.encoder(face_image)  # type: ignore[no-any-return]


def create_identity_decoder(
    base_decoder: WanDecoder3d,
    face_embedding_dim: int = 512,
) -> WanDecoder3dWithIdentity:
    """Create a `.wreel` world from a `.wscript` scene."""
    return WanDecoder3dWithIdentity(base_decoder, face_embedding_dim=face_embedding_dim)
