"""
DiT Student Model - Causal World Model
Autoregressive diffusion transformer for world modeling with KV caching.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple, List
import math
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from blocks.dit_block import DiTBlock
from layers.rope import RoPELayer, precompute_rope_3d_freqs
from layers.nn_helpers import AdaLN, RMSNorm, PositionalEncoding
from blocks.flow_match import FlowMatchingSampler


class TokenEmbedding(nn.Module):
    """
    Token embedding for different input modalities.
    Handles RGB tokens, depth tokens, memory tokens, and control tokens.
    """
    
    def __init__(
        self,
        latent_dim: int,
        d_model: int,
        vocab_size: Optional[int] = None,
        max_seq_len: int = 2048
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.d_model = d_model
        self.vocab_size = vocab_size
        
        # Latent token projection (for VAE latents)
        self.latent_proj = nn.Linear(latent_dim, d_model)
        
        # Discrete token embedding (if using quantized tokens)
        if vocab_size is not None:
            self.token_embed = nn.Embedding(vocab_size, d_model)
        
        # Control signal embedding
        self.control_proj = nn.Linear(4, d_model)  # steer, accel, goal_x, goal_y
        
        # Modality embeddings
        self.modality_embed = nn.Embedding(4, d_model)  # rgb, depth, memory, control
        
        # Position embedding
        self.pos_embed = PositionalEncoding(d_model, max_seq_len)
        
    def forward(
        self,
        latent_tokens: Optional[torch.Tensor] = None,
        discrete_tokens: Optional[torch.Tensor] = None,
        control_signals: Optional[torch.Tensor] = None,
        modality_ids: Optional[torch.Tensor] = None,
        add_pos_embed: bool = True
    ) -> torch.Tensor:
        """
        Embed different types of tokens.
        
        Args:
            latent_tokens: Continuous latent tokens [B, T, latent_dim]
            discrete_tokens: Discrete token IDs [B, T]
            control_signals: Control signals [B, T, 4]
            modality_ids: Modality type IDs [B, T]
            add_pos_embed: Whether to add positional embeddings
        
        Returns:
            Embedded tokens [B, T, d_model]
        """
        tokens = []
        
        # Process latent tokens
        if latent_tokens is not None:
            tokens.append(self.latent_proj(latent_tokens))
        
        # Process discrete tokens
        if discrete_tokens is not None:
            tokens.append(self.token_embed(discrete_tokens))
        
        # Process control signals
        if control_signals is not None:
            tokens.append(self.control_proj(control_signals))
        
        # Combine all tokens
        if len(tokens) == 1:
            embedded = tokens[0]
        else:
            embedded = torch.cat(tokens, dim=1)
        
        # Add modality embeddings
        if modality_ids is not None:
            embedded = embedded + self.modality_embed(modality_ids)
        
        # Add positional embeddings
        if add_pos_embed:
            embedded = self.pos_embed(embedded)
        
        return embedded


class MemoryBank(nn.Module):
    """
    Memory bank for storing and retrieving relevant context.
    Implements attention-based memory for spatial/object permanence.
    """
    
    def __init__(
        self,
        d_model: int,
        max_memory_size: int = 1024,
        num_heads: int = 8,
        memory_decay: float = 0.99
    ):
        super().__init__()
        self.d_model = d_model
        self.max_memory_size = max_memory_size
        self.num_heads = num_heads
        self.memory_decay = memory_decay
        
        self.memory_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            batch_first=True
        )
        
        self.memory_proj = nn.Linear(d_model, d_model)
        self.gate = nn.Linear(2 * d_model, d_model)
        
        # Initialize empty memory
        self.register_buffer('memory', torch.zeros(1, 0, d_model))
        self.register_buffer('memory_scores', torch.zeros(1, 0))
        
    def update(self, new_tokens: torch.Tensor, importance_scores: Optional[torch.Tensor] = None):
        """
        Update memory bank with new tokens.
        
        Args:
            new_tokens: New tokens to add [B, T, d_model]
            importance_scores: Importance scores for memory retention [B, T]
        """
        B, T, D = new_tokens.shape
        
        if importance_scores is None:
            importance_scores = torch.ones(B, T, device=new_tokens.device)
        
        # Decay existing memory scores
        self.memory_scores *= self.memory_decay
        
        # Add new tokens to memory
        self.memory = torch.cat([self.memory, new_tokens], dim=1)
        self.memory_scores = torch.cat([self.memory_scores, importance_scores], dim=1)
        
        # Prune memory if too large
        if self.memory.size(1) > self.max_memory_size:
            # Keep top-k most important memories
            _, top_indices = torch.topk(
                self.memory_scores, 
                k=self.max_memory_size, 
                dim=1
            )
            
            # Sort indices to maintain temporal order
            top_indices = torch.sort(top_indices, dim=1)[0]
            
            self.memory = torch.gather(
                self.memory, 1, 
                top_indices.unsqueeze(-1).expand(-1, -1, D)
            )
            self.memory_scores = torch.gather(self.memory_scores, 1, top_indices)
    
    def retrieve(self, query: torch.Tensor, top_k: int = 64) -> torch.Tensor:
        """
        Retrieve relevant memories using attention.
        
        Args:
            query: Query tokens [B, T, d_model]
            top_k: Number of top memories to retrieve
        
        Returns:
            Retrieved memory tokens [B, top_k, d_model]
        """
        if self.memory.size(1) == 0:
            # No memories available
            return torch.zeros(query.size(0), top_k, self.d_model, device=query.device)
        
        # Compute attention scores
        attn_output, attn_weights = self.memory_attn(
            query=query,
            key=self.memory,
            value=self.memory
        )
        
        # Weight by importance scores
        weighted_attn = attn_weights * self.memory_scores.unsqueeze(1)
        
        # Select top-k memories
        _, top_indices = torch.topk(weighted_attn.mean(1), k=min(top_k, self.memory.size(1)), dim=1)
        
        retrieved = torch.gather(
            self.memory, 1,
            top_indices.unsqueeze(-1).expand(-1, -1, self.d_model)
        )
        
        return retrieved


class DiTStudent(nn.Module):
    """
    DiT Student model for causal world modeling.
    Autoregressive transformer with KV caching for efficient inference.
    """
    
    def __init__(
        self,
        latent_dim: int = 8,
        d_model: int = 1024,
        n_layers: int = 24,
        n_heads: int = 16,
        d_ff: Optional[int] = None,
        mlp_type: str = 'swiglu',
        dropout: float = 0.0,
        max_seq_len: int = 2048,
        vocab_size: Optional[int] = None,
        use_rope: bool = True,
        use_memory: bool = True,
        memory_size: int = 1024,
        bias: bool = False
    ):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.max_seq_len = max_seq_len
        self.use_rope = use_rope
        self.use_memory = use_memory
        
        # Token embedding
        self.token_embed = TokenEmbedding(
            latent_dim=latent_dim,
            d_model=d_model,
            vocab_size=vocab_size,
            max_seq_len=max_seq_len
        )
        
        # RoPE for 3D (time, height, width) if enabled
        if use_rope:
            self.rope_layer = RoPELayer(
                dim=d_model // n_heads,
                max_seq_len=max_seq_len
            )
        else:
            self.rope_layer = None
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            DiTBlock(
                d_model=d_model,
                n_heads=n_heads,
                d_ff=d_ff,
                mlp_type=mlp_type,
                dropout=dropout,
                causal=True,  # Causal for student
                use_rope=use_rope,
                max_seq_len=max_seq_len,
                bias=bias
            )
            for _ in range(n_layers)
        ])
        
        # Final layer norm
        self.norm = RMSNorm(d_model)
        
        # Output heads
        self.lm_head = nn.Linear(d_model, latent_dim, bias=False)  # For latent prediction
        
        # Memory bank
        if use_memory:
            self.memory_bank = MemoryBank(
                d_model=d_model,
                max_memory_size=memory_size,
                num_heads=n_heads
            )
        else:
            self.memory_bank = None
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        """Initialize weights with scaled normal distribution."""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.02)
    
    def forward(
        self,
        tokens: torch.Tensor,
        kv_cache: Optional[List[Dict[str, torch.Tensor]]] = None,
        use_cache: bool = False,
        causal_mask: Optional[torch.Tensor] = None,
        memory_context: Optional[torch.Tensor] = None,
        return_hidden: bool = False
    ) -> Tuple[torch.Tensor, Optional[List[Dict[str, torch.Tensor]]]]:
        """
        Forward pass through DiT student.
        
        Args:
            tokens: Input tokens [B, T, latent_dim] or token IDs [B, T]
            kv_cache: List of KV caches for each layer
            use_cache: Whether to use/return KV cache
            causal_mask: Optional causal mask override
            memory_context: Optional memory context from memory bank
            return_hidden: Whether to return hidden states
        
        Returns:
            Tuple of (output, kv_cache) where output is [B, T, latent_dim]
        """
        B, T = tokens.shape[:2]
        device = tokens.device
        
        # Embed tokens
        if tokens.dim() == 2:  # Discrete tokens
            x = self.token_embed(discrete_tokens=tokens)
        else:  # Continuous tokens
            x = self.token_embed(latent_tokens=tokens)
        
        # Initialize KV cache if needed
        if use_cache and kv_cache is None:
            kv_cache = [None] * self.n_layers
        
        # Add memory context if available
        if memory_context is not None:
            # Prepend memory context to input
            x = torch.cat([memory_context, x], dim=1)
            
            # Adjust mask for memory context
            if causal_mask is not None:
                mem_len = memory_context.size(1)
                mem_mask = torch.ones(B, 1, T, mem_len, device=device)
                causal_mask = torch.cat([mem_mask, causal_mask], dim=-1)
        
        # Forward through transformer blocks
        new_kv_cache = []
        hidden_states = []
        
        for i, block in enumerate(self.blocks):
            layer_kv_cache = kv_cache[i] if kv_cache is not None else None
            
            x, new_layer_kv_cache = block(
                x=x,
                kv_cache=layer_kv_cache,
                causal_mask=causal_mask
            )
            
            if use_cache:
                new_kv_cache.append(new_layer_kv_cache)
            
            if return_hidden:
                hidden_states.append(x)
        
        # Final layer norm
        x = self.norm(x)
        
        # If we added memory context, remove it from output
        if memory_context is not None:
            mem_len = memory_context.size(1)
            x = x[:, mem_len:]
        
        # Output projection
        output = self.lm_head(x)
        
        result = output
        if return_hidden:
            result = (output, hidden_states)
        
        if use_cache:
            return result, new_kv_cache
        else:
            return result, None
    
    def generate(
        self,
        context: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        use_memory: bool = True
    ) -> torch.Tensor:
        """
        Autoregressive generation with KV caching.
        
        Args:
            context: Context tokens [B, T_ctx, latent_dim]
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling
            top_p: Top-p (nucleus) sampling
            use_memory: Whether to use memory bank
        
        Returns:
            Generated sequence [B, T_ctx + max_new_tokens, latent_dim]
        """
        B, T_ctx, D = context.shape
        device = context.device
        
        # Initialize sequence with context
        sequence = context
        kv_cache = None
        
        # Initialize memory bank for this generation
        if use_memory and self.memory_bank is not None:
            # Add context to memory
            self.memory_bank.update(context)
        
        for _ in range(max_new_tokens):
            # Get current token (last token for KV caching)
            if kv_cache is None:
                # First forward pass - use full sequence
                current_tokens = sequence
            else:
                # Subsequent passes - only last token
                current_tokens = sequence[:, -1:, :]
            
            # Retrieve memory context if enabled
            memory_context = None
            if use_memory and self.memory_bank is not None:
                memory_context = self.memory_bank.retrieve(current_tokens, top_k=64)
            
            # Forward pass
            logits, kv_cache = self.forward(
                tokens=current_tokens,
                kv_cache=kv_cache,
                use_cache=True,
                memory_context=memory_context
            )
            
            # Get logits for last position
            next_token_logits = logits[:, -1, :]  # [B, latent_dim]
            
            # Apply temperature
            if temperature != 1.0:
                next_token_logits = next_token_logits / temperature
            
            # Sample next token (using continuous sampling for latent space)
            if top_k is not None or top_p is not None:
                # For continuous latents, we could use mixture of Gaussians or other methods
                # For simplicity, using direct sampling with noise
                next_token = next_token_logits + torch.randn_like(next_token_logits) * temperature
            else:
                next_token = next_token_logits + torch.randn_like(next_token_logits) * temperature
            
            next_token = next_token.unsqueeze(1)  # [B, 1, latent_dim]
            
            # Append to sequence
            sequence = torch.cat([sequence, next_token], dim=1)
            
            # Update memory bank
            if use_memory and self.memory_bank is not None:
                self.memory_bank.update(next_token)
        
        return sequence
    
    def compute_loss(
        self,
        input_tokens: torch.Tensor,
        target_tokens: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute autoregressive loss.
        
        Args:
            input_tokens: Input tokens [B, T, latent_dim]
            target_tokens: Target tokens [B, T, latent_dim]
            mask: Optional mask for valid positions [B, T]
        
        Returns:
            Loss scalar
        """
        # Forward pass
        pred_tokens, _ = self.forward(input_tokens)
        
        # Compute loss (MSE for continuous latents)
        loss = F.mse_loss(pred_tokens, target_tokens, reduction='none')
        
        # Apply mask if provided
        if mask is not None:
            loss = loss * mask.unsqueeze(-1)
            loss = loss.sum() / mask.sum()
        else:
            loss = loss.mean()
        
        return loss


def test_dit_student():
    """Test function for DiT Student."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create model
    model = DiTStudent(
        latent_dim=8,
        d_model=512,  # Smaller for testing
        n_layers=6,   # Fewer layers for testing
        n_heads=8,
        max_seq_len=128,
        use_rope=True,
        use_memory=True
    ).to(device)
    
    # Test input
    B, T, D = 2, 32, 8
    tokens = torch.randn(B, T, D).to(device)
    
    print(f"Input shape: {tokens.shape}")
    
    # Test forward pass
    output, kv_cache = model(tokens, use_cache=True)
    print(f"Output shape: {output.shape}")
    print(f"KV cache length: {len(kv_cache) if kv_cache else 0}")
    
    # Test generation
    context = tokens[:, :16, :]  # Use first 16 tokens as context
    generated = model.generate(context, max_new_tokens=16, temperature=0.8)
    print(f"Generated shape: {generated.shape}")
    
    # Test loss computation
    target = torch.randn(B, T, D).to(device)
    loss = model.compute_loss(tokens, target)
    print(f"Loss: {loss.item():.4f}")


if __name__ == "__main__":
    test_dit_student()