"""
Head modules for the router action layer.
Supports different architectures: linear, low-rank, sparse, block-diagonal, turn-aware.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional, Dict


class RouterHead(nn.Module):
    """
    Flexible router head that supports different architectures.
    """

    def __init__(
            self,
            hidden_size: int,
            num_agents: int,
            head_type: str = "linear",
            max_turns: int = 5,
            device: str = "cuda:0",
            dtype: torch.dtype = torch.bfloat16,
            debug: bool = False
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_agents = num_agents
        self.head_type = head_type
        self.max_turns = max_turns
        self.device = device
        self.dtype = dtype
        self.debug = debug

        # FIXED: Initialize turn embedding cache as instance variable
        self._turn_embeddings_cache = {}
        self._forward_count = 0

        if self.debug:
            print(f"[RouterHead] Initializing {head_type} head:")
            print(f"  Hidden size: {hidden_size}")
            print(f"  Num agents: {num_agents}")
            print(f"  Max turns: {max_turns}")
            print(f"  Device: {device}")
            print(f"  Dtype: {dtype}")

        if head_type == "linear":
            self._init_linear()
        elif head_type == "low-rank":
            self._init_low_rank()
        elif head_type == "sparse":
            self._init_sparse()
        elif head_type == "block-diagonal":
            self._init_block_diagonal()
        elif head_type == "turn":
            self._init_turn_aware()
        else:
            raise ValueError(f"Unknown head_type: {head_type}")

        # Calculate and debug parameter count
        param_count = self.get_parameter_count()
        if self.debug:
            print(f"[RouterHead] {head_type} head initialized with {param_count} parameters")
            self._debug_parameter_breakdown()

    def _debug_parameter_breakdown(self):
        """Print detailed parameter breakdown for debugging."""
        if self.head_type == "linear":
            print(
                f"  Linear layer: {self.hidden_size} × {self.num_agents} = {self.hidden_size * self.num_agents} params")
        elif self.head_type == "low-rank":
            u_params = self.rank * self.hidden_size
            v_params = self.rank * self.num_agents
            print(f"  U layer: {self.hidden_size} × {self.rank} = {u_params} params")
            print(f"  V layer: {self.rank} × {self.num_agents} = {v_params} params")
            print(f"  Total: {u_params + v_params} params (fixed scale not optimized)")
        elif self.head_type == "sparse":
            linear_params = self.hidden_size * self.num_agents
            adaptive_params = self.hidden_size + 2
            print(f"  Linear layer: {self.hidden_size} × {self.num_agents} = {linear_params} params")
            print(f"  Dimension scores: {self.hidden_size} params")
            print(f"  Adaptive params: 2 params (temp, sparsity)")
            print(f"  Total: {linear_params + adaptive_params} params")
        elif self.head_type == "block-diagonal":
            total = 0
            for i, block in enumerate(self.blocks):
                block_params = block.weight.numel()
                total += block_params
                print(
                    f"  Block {i}: {self.hidden_distribution[i]} × {self.agent_distribution[i]} = {block_params} params")
            print(f"  Total: {total} params")
        elif self.head_type == "turn":
            # FIXED: Only count linear layer parameters (turn embeddings are frozen)
            combined_input_size = self.hidden_size + self.turn_embedding_dim
            linear_params = combined_input_size * self.num_agents
            print(f"  Turn embedding: ({self.max_turns} + 1) × {self.turn_embedding_dim} = FROZEN (not optimized)")
            print(
                f"  Combined linear: ({self.hidden_size} + {self.turn_embedding_dim}) × {self.num_agents} = {linear_params} params")
            print(f"  Total optimizable: {linear_params} params")
            print(
                f"  Turn embedding dim calculation: max(64, {self.hidden_size}*0.1) = {self.turn_embedding_dim}")

    def _init_linear(self):
        """Standard linear layer."""
        self.linear = nn.Linear(self.hidden_size, self.num_agents, bias=False)
        self.linear = self.linear.to(self.device).to(self.dtype)

        if self.debug:
            print(f"[RouterHead] Linear layer shape: {self.linear.weight.shape}")

    def _init_low_rank(self):
        """Low-rank factorization: hidden -> r -> agents."""
        # FIXED: More reasonable rank calculation
        min_rank = max(8, self.num_agents * 2)  # At least 2x num_agents
        max_rank = min(self.hidden_size // 4, self.num_agents * 32)  # Reasonable upper bound

        # Target 70-80% of original parameters for meaningful compression
        target_params = int(0.75 * self.hidden_size * self.num_agents)
        calculated_r = int(target_params / (self.hidden_size + self.num_agents))

        # Clamp to reasonable range
        self.rank = max(min_rank, min(max_rank, calculated_r))

        if self.debug:
            print(f"[RouterHead] Low-rank calculation:")
            print(f"  Target params: {target_params}")
            print(f"  Calculated r: {calculated_r} -> Clamped r: {self.rank}")
            print(f"  Actual params: {self.rank * (self.hidden_size + self.num_agents)}")
            print(
                f"  Compression ratio: {(self.rank * (self.hidden_size + self.num_agents)) / (self.hidden_size * self.num_agents):.3f}")

        # Better initialization for low-rank layers
        self.U = nn.Linear(self.hidden_size, self.rank, bias=False)
        self.V = nn.Linear(self.rank, self.num_agents, bias=False)

        # Initialize with Xavier/Glorot scaling for better signal propagation
        nn.init.xavier_uniform_(self.U.weight, gain=1.0)
        nn.init.xavier_uniform_(self.V.weight, gain=3.0)  # Higher gain for stronger signals

        self.U = self.U.to(self.device).to(self.dtype)
        self.V = self.V.to(self.device).to(self.dtype)

        # FIXED: Use fixed scaling instead of learnable (remove from optimization)
        self.register_buffer('output_scale', torch.tensor(3.0, device=self.device, dtype=self.dtype))

        if self.debug:
            print(f"  U layer shape: {self.U.weight.shape}")
            print(f"  V layer shape: {self.V.weight.shape}")
            print(f"  Fixed output scale: {self.output_scale.item():.3f}")

    def _init_sparse(self):
        """Sparse linear layer with learnable dimension selection."""
        if self.debug:
            print(f"[RouterHead] Initializing adaptive sparse head")

        # Better initialization: larger variance for dimension scores
        self.dimension_scores = nn.Parameter(
            torch.randn(self.hidden_size, device=self.device, dtype=self.dtype) * 2.0
        )

        # Higher initial temperature for gentler selection
        self.selection_temperature = nn.Parameter(
            torch.tensor(10.0, device=self.device, dtype=self.dtype)
        )

        # More conservative sparsity (start less sparse)
        self.sparsity_logit = nn.Parameter(
            torch.tensor(0.0, device=self.device, dtype=self.dtype)  # sigmoid(0) = 0.5
        )

        # Linear layer that operates on all dimensions (will be soft-masked)
        self.linear = nn.Linear(self.hidden_size, self.num_agents, bias=False)
        self.linear = self.linear.to(self.device).to(self.dtype)

        if self.debug:
            print(f"  Dimension scores shape: {self.dimension_scores.shape}")
            print(
                f"  Dimension scores range: [{self.dimension_scores.min().item():.3f}, {self.dimension_scores.max().item():.3f}]")
            print(f"  Linear layer shape: {self.linear.weight.shape}")
            print(f"  Initial sparsity ratio: {torch.sigmoid(self.sparsity_logit).item():.3f}")
            print(f"  Initial temperature: {self.selection_temperature.item():.3f}")

    def _init_block_diagonal(self):
        """Block-diagonal structure with proportional hidden dimension allocation."""
        if self.debug:
            print(f"[RouterHead] Initializing block-diagonal head")

        # Partition agents into blocks
        agents_per_block = max(1, self.num_agents // 2)  # 2 blocks by default
        num_blocks = (self.num_agents + agents_per_block - 1) // agents_per_block

        # Calculate agent distribution for each block
        agent_distribution = []
        remaining_agents = self.num_agents
        for i in range(num_blocks):
            agents_in_this_block = min(agents_per_block, remaining_agents)
            agent_distribution.append(agents_in_this_block)
            remaining_agents -= agents_in_this_block

        # Allocate hidden dimensions proportionally to agent count
        hidden_distribution = []
        remaining_hidden = self.hidden_size
        for i, agents_in_block in enumerate(agent_distribution):
            if i == len(agent_distribution) - 1:  # Last block gets remaining dims
                hidden_for_block = remaining_hidden
            else:
                # Proportional allocation: (agents_in_block / total_agents) * hidden_size
                hidden_for_block = (agents_in_block * self.hidden_size) // self.num_agents
                remaining_hidden -= hidden_for_block
            hidden_distribution.append(hidden_for_block)

        # Store configuration
        self.num_blocks = num_blocks
        self.agent_distribution = agent_distribution
        self.hidden_distribution = hidden_distribution

        if self.debug:
            print(f"  Number of blocks: {num_blocks}")
            print(f"  Agent distribution: {agent_distribution}")
            print(f"  Hidden distribution: {hidden_distribution}")

        # Create blocks with proportional sizes
        self.blocks = nn.ModuleList()
        for i in range(num_blocks):
            block = nn.Linear(
                hidden_distribution[i],
                agent_distribution[i],
                bias=False
            )
            self.blocks.append(block.to(self.device).to(self.dtype))

            if self.debug:
                dims_per_agent = hidden_distribution[i] / agent_distribution[i]
                print(f"  Block {i}: {hidden_distribution[i]} dims → {agent_distribution[i]} agents "
                      f"({dims_per_agent:.1f} dims/agent, shape: {block.weight.shape})")

    def _init_turn_aware(self):
        """Turn-aware linear layer with turn embedding."""
        if self.debug:
            print(f"[RouterHead] Initializing turn-aware head")

        # Turn embedding dimension (can be tuned)
        self.turn_embedding_dim = max(64, int(self.hidden_size * 0.1))

        # FIXED: Turn embedding layer with better initialization
        self.turn_embedding = nn.Embedding(
            self.max_turns + 1,  # +1 to handle turn 0
            self.turn_embedding_dim
        )

        # FIXED: Initialize with deterministic positional encoding-like pattern
        with torch.no_grad():
            # Use sinusoidal positional encoding initialization for better turn representation
            for turn_idx in range(self.max_turns + 1):
                for dim_idx in range(self.turn_embedding_dim):
                    if dim_idx % 2 == 0:
                        # Even dimensions: sine
                        self.turn_embedding.weight[turn_idx, dim_idx] = np.sin(
                            turn_idx / (10000 ** (dim_idx / self.turn_embedding_dim))
                        )
                    else:
                        # Odd dimensions: cosine
                        self.turn_embedding.weight[turn_idx, dim_idx] = np.cos(
                            turn_idx / (10000 ** ((dim_idx - 1) / self.turn_embedding_dim))
                        )

        # FIXED: Freeze turn embedding weights (exclude from optimization)
        for param in self.turn_embedding.parameters():
            param.requires_grad = False

        # Combined input size: hidden states + turn embedding
        combined_input_size = self.hidden_size + self.turn_embedding_dim

        # Linear layer with combined input
        self.linear = nn.Linear(combined_input_size, self.num_agents, bias=False)

        # Move to device and set dtype
        self.turn_embedding = self.turn_embedding.to(self.device).to(self.dtype)
        self.linear = self.linear.to(self.device).to(self.dtype)

        if self.debug:
            print(f"  Turn embedding dim: {self.turn_embedding_dim}")
            print(f"  Combined input size: {combined_input_size}")
            print(f"  Linear layer shape: {self.linear.weight.shape}")
            print(f"  Turn embedding shape: {self.turn_embedding.weight.shape}")
            print(f"  Turn embedding frozen: {not self.turn_embedding.weight.requires_grad}")

    def forward(self, x: torch.Tensor, turn_num: int = 0) -> torch.Tensor:
        """Forward pass through the head."""
        self._forward_count += 1

        if self.debug and self._forward_count <= 3:
            print(
                f"[RouterHead] Forward pass #{self._forward_count}, input shape: {x.shape}, dtype: {x.dtype}, turn: {turn_num}")

        if self.head_type == "linear":
            result = self.linear(x)
        elif self.head_type == "low-rank":
            u_out = self.U(x)
            if self.debug and self._forward_count <= 3:
                print(
                    f"  U output shape: {u_out.shape}, mean: {u_out.mean().item():.6f}, std: {u_out.std().item():.6f}")

            elu_out = torch.nn.functional.elu(u_out, alpha=0.1)
            v_out = self.V(elu_out)
            result = v_out * self.output_scale

            if self.debug and self._forward_count <= 3:
                print(f"  ELU output mean: {elu_out.mean().item():.6f}, std: {elu_out.std().item():.6f}")
                print(f"  V output mean: {v_out.mean().item():.6f}, std: {v_out.std().item():.6f}")
                print(f"  Output scale: {self.output_scale.item():.3f}")

        elif self.head_type == "sparse":
            result = self._forward_adaptive_sparse(x)
        elif self.head_type == "block-diagonal":
            outputs = []
            hidden_offset = 0
            for i, block in enumerate(self.blocks):
                hidden_size_for_block = self.hidden_distribution[i]
                x_block = x[..., hidden_offset:hidden_offset + hidden_size_for_block]
                block_out = block(x_block)
                outputs.append(block_out)
                if self.debug and self._forward_count <= 3:
                    print(f"  Block {i} input shape: {x_block.shape}, output shape: {block_out.shape}")
                hidden_offset += hidden_size_for_block
            result = torch.cat(outputs, dim=-1)
        elif self.head_type == "turn":
            result = self._forward_turn_aware(x, turn_num)
        else:
            raise ValueError(f"Unknown head_type: {self.head_type}")

        if self.debug and self._forward_count <= 3:
            print(
                f"  Final output shape: {result.shape}, mean: {result.mean().item():.6f}, std: {result.std().item():.6f}")
            print(f"  Output range: [{result.min().item():.6f}, {result.max().item():.6f}]")

        return result

    def _forward_adaptive_sparse(self, x: torch.Tensor) -> torch.Tensor:
        """Simplified sparse forward with top-k selection."""
        # Ensure input has same dtype as linear layer
        target_dtype = self.linear.weight.dtype
        x = x.to(dtype=target_dtype)

        # Compute current sparsity target
        sparsity_ratio = torch.sigmoid(self.sparsity_logit)
        target_k = max(1, int(self.hidden_size * (1 - sparsity_ratio)))

        if self.debug and self._forward_count <= 3:
            print(f"  Sparse forward: sparsity_ratio={sparsity_ratio.item():.3f}, target_k={target_k}")
            print(
                f"  Dimension scores range: [{self.dimension_scores.min().item():.3f}, {self.dimension_scores.max().item():.3f}]")

        if self.training:
            # Training: Use Gumbel-Softmax for differentiable top-k selection
            temperature = torch.clamp(self.selection_temperature, min=1.0, max=20.0)

            # Add Gumbel noise for exploration
            gumbel_noise = -torch.log(-torch.log(torch.rand_like(self.dimension_scores) + 1e-8) + 1e-8)
            noisy_scores = (self.dimension_scores + gumbel_noise) / temperature

            # Get top-k through differentiable sampling
            _, top_indices = torch.topk(noisy_scores, target_k)

            # Create soft mask that focuses on top-k dimensions
            soft_mask = torch.zeros_like(self.dimension_scores, dtype=target_dtype)
            soft_mask[top_indices] = 1.0

            # Apply soft selection with slight regularization toward selected dimensions
            attention_weights = torch.softmax(self.dimension_scores / temperature, dim=0)
            final_weights = attention_weights * soft_mask

            # Normalize to maintain scale
            final_weights = final_weights / (final_weights.sum() + 1e-8) * target_k

            if self.debug and self._forward_count <= 3:
                print(f"  Training: temp={temperature.item():.3f}, active_dims={soft_mask.sum().item()}")
                print(f"  Final weights range: [{final_weights.min().item():.6f}, {final_weights.max().item():.6f}]")
                print(f"  Final weights sum: {final_weights.sum().item():.3f}")

        else:
            # Inference: Hard top-k selection
            _, top_indices = torch.topk(self.dimension_scores, target_k)
            final_weights = torch.zeros_like(self.dimension_scores, dtype=target_dtype)
            final_weights[top_indices] = 1.0

            if self.debug and self._forward_count <= 3:
                print(f"  Inference: selected {target_k} dimensions")
                print(f"  Selected indices (first 10): {top_indices[:10].tolist()}")

        # Apply selection and pass through linear layer
        x_masked = x * final_weights.unsqueeze(0)
        return self.linear(x_masked)

    def _forward_turn_aware(self, x: torch.Tensor, turn_num: int) -> torch.Tensor:
        """Forward pass for turn-aware head with fixed debugging."""
        batch_size = x.shape[0]

        # Clamp turn number to valid range
        turn_clamped = min(max(turn_num, 0), self.max_turns)

        # Create turn tensor
        turn_tensor = torch.tensor([turn_clamped], device=x.device, dtype=torch.long)
        turn_embed = self.turn_embedding(turn_tensor)  # [1, turn_embedding_dim]

        # Expand turn embedding to match batch size
        turn_embed = turn_embed.expand(batch_size, -1)  # [batch_size, turn_embedding_dim]

        # FIXED: Enhanced turn embedding debugging with proper caching
        if self.debug and self._forward_count <= 10:
            print(f"  Turn-aware forward: turn_num={turn_num}, clamped={turn_clamped}")
            print(f"  Turn embedding shape: {turn_embed.shape}")
            print(f"  Turn embedding mean: {turn_embed.mean().item():.6f}")
            print(f"  Turn embedding std: {turn_embed.std().item():.6f}")

            # Convert to float32 for numpy operations
            current_embedding = turn_embed[0].detach().cpu().float().numpy()
            turn_key = f"turn_{turn_clamped}"

            # Check if this turn has been seen before (proper caching)
            if turn_key in self._turn_embeddings_cache:
                # Compare with previous embedding for same turn
                prev_embedding = self._turn_embeddings_cache[turn_key]
                embedding_diff = np.abs(current_embedding - prev_embedding).mean()

                print(f"  Turn {turn_clamped} embedding consistency check:")
                print(f"    Mean absolute difference from previous: {embedding_diff:.8f}")

                if embedding_diff > 1e-6:
                    print(f"    ERROR: Turn embeddings changed! This indicates optimization bug.")
                    print(f"    Previous sample: {prev_embedding[:5]}")
                    print(f"    Current sample:  {current_embedding[:5]}")
                else:
                    print(f"    ✓ Turn embeddings are consistent")
            else:
                # First time seeing this turn
                print(f"  First occurrence of turn {turn_clamped}")
                print(f"  Turn embedding sample values: {current_embedding[:5]}")

                # Store current embedding (FIXED: proper caching)
                self._turn_embeddings_cache[turn_key] = current_embedding.copy()

            # Compare with other turns if we have them
            if len(self._turn_embeddings_cache) > 1:
                print(f"  Turn embedding comparisons:")
                for other_turn_key, other_embedding in self._turn_embeddings_cache.items():
                    if other_turn_key != turn_key:
                        other_turn_num = int(other_turn_key.split('_')[1])
                        turn_diff = np.abs(current_embedding - other_embedding).mean()
                        cosine_sim = np.dot(current_embedding, other_embedding) / (
                                np.linalg.norm(current_embedding) * np.linalg.norm(other_embedding)
                        )
                        print(
                            f"    vs Turn {other_turn_num}: mean_abs_diff={turn_diff:.6f}, cosine_sim={cosine_sim:.6f}")

                # Summary statistics
                all_turns = sorted([int(k.split('_')[1]) for k in self._turn_embeddings_cache.keys()])
                print(f"  Cached turn embeddings: {all_turns}")

        # Concatenate hidden states with turn embedding
        x_combined = torch.cat([x, turn_embed], dim=-1)

        # Pass through linear layer
        result = self.linear(x_combined)

        if self.debug and self._forward_count <= 3:
            print(f"  Combined input shape: {x_combined.shape}")
            print(f"  Linear output shape: {result.shape}")

        return result

    def get_parameter_count(self) -> int:
        """Get total number of optimizable parameters in this head."""
        if self.head_type == "linear":
            count = self.hidden_size * self.num_agents
        elif self.head_type == "low-rank":
            count = self.rank * (self.hidden_size + self.num_agents)
        elif self.head_type == "sparse":
            linear_params = self.hidden_size * self.num_agents
            adaptive_params = self.hidden_size + 2
            count = linear_params + adaptive_params
        elif self.head_type == "block-diagonal":
            count = 0
            for block in self.blocks:
                count += block.weight.numel()
        elif self.head_type == "turn":
            # FIXED: Only count linear layer parameters (turn embeddings are frozen)
            combined_input_size = self.hidden_size + self.turn_embedding_dim
            count = combined_input_size * self.num_agents
        else:
            count = 0

        if self.debug:
            print(f"[RouterHead] Optimizable parameter count for {self.head_type}: {count}")

        return count

    def get_weight_tensor(self) -> torch.Tensor:
        """Get flattened weight tensor for CMA-ES optimization (FIXED: excludes turn embeddings)."""
        if self.debug:
            print(f"[RouterHead] Getting weight tensor for {self.head_type}")

        if self.head_type == "linear":
            tensor = self.linear.weight.flatten()
        elif self.head_type == "low-rank":
            u_weights = self.U.weight.flatten()
            v_weights = self.V.weight.flatten()
            tensor = torch.cat([u_weights, v_weights])
            if self.debug:
                print(f"  U weights: {u_weights.shape}, V weights: {v_weights.shape}")
                print(f"  Fixed scale: {self.output_scale.item():.3f} (not optimized)")
        elif self.head_type == "sparse":
            linear_weights = self.linear.weight.flatten()
            scores = self.dimension_scores.flatten()
            temp = self.selection_temperature.unsqueeze(0)
            sparsity = self.sparsity_logit.unsqueeze(0)
            tensor = torch.cat([linear_weights, scores, temp, sparsity])
            if self.debug:
                print(f"  Linear: {linear_weights.shape}, Scores: {scores.shape}")
                print(f"  Adaptive params: temp={temp.item():.3f}, sparsity={sparsity.item():.3f}")
        elif self.head_type == "block-diagonal":
            weights = []
            for i, block in enumerate(self.blocks):
                block_weights = block.weight.flatten()
                weights.append(block_weights)
                if self.debug:
                    print(f"  Block {i} weights: {block_weights.shape}")
            tensor = torch.cat(weights)
        elif self.head_type == "turn":
            # FIXED: Only include linear layer weights (exclude turn embeddings)
            linear_weights = self.linear.weight.flatten()
            tensor = linear_weights
            if self.debug:
                print(f"  Linear weights only: {linear_weights.shape}")
                print(f"  Turn embeddings excluded (frozen)")

        if self.debug:
            print(f"  Final tensor shape: {tensor.shape}, dtype: {tensor.dtype}")

        return tensor

    def set_weight_tensor(self, weight_tensor: torch.Tensor):
        """Set weights from flattened tensor for CMA-ES optimization (FIXED: excludes turn embeddings)."""
        if self.debug:
            print(f"[RouterHead] Setting weight tensor for {self.head_type}, input shape: {weight_tensor.shape}")

        offset = 0

        if self.head_type == "linear":
            weight_size = self.linear.weight.numel()
            if self.debug:
                print(f"  Setting linear weights: {weight_size} elements")
            self.linear.weight.data.copy_(
                weight_tensor[offset:offset + weight_size].view_as(self.linear.weight)
            )
        elif self.head_type == "low-rank":
            u_size = self.U.weight.numel()
            v_size = self.V.weight.numel()

            if self.debug:
                print(f"  Setting U weights: {u_size} elements at offset {offset}")
            self.U.weight.data.copy_(
                weight_tensor[offset:offset + u_size].view_as(self.U.weight)
            )
            offset += u_size

            if self.debug:
                print(f"  Setting V weights: {v_size} elements at offset {offset}")
            self.V.weight.data.copy_(
                weight_tensor[offset:offset + v_size].view_as(self.V.weight)
            )
        elif self.head_type == "sparse":
            linear_weight_size = self.linear.weight.numel()
            if self.debug:
                print(f"  Setting linear weights: {linear_weight_size} elements at offset {offset}")
            self.linear.weight.data.copy_(
                weight_tensor[offset:offset + linear_weight_size].view_as(self.linear.weight)
            )
            offset += linear_weight_size

            scores_size = self.dimension_scores.numel()
            if self.debug:
                print(f"  Setting dimension scores: {scores_size} elements at offset {offset}")
            self.dimension_scores.data.copy_(
                weight_tensor[offset:offset + scores_size].view_as(self.dimension_scores)
            )
            offset += scores_size

            if self.debug:
                print(f"  Setting adaptive params at offset {offset}")
                print(f"    temp: {weight_tensor[offset].item():.3f}")
                print(f"    sparsity: {weight_tensor[offset + 1].item():.3f}")

            self.selection_temperature.data.copy_(weight_tensor[offset])
            offset += 1
            self.sparsity_logit.data.copy_(weight_tensor[offset])
        elif self.head_type == "block-diagonal":
            for i, block in enumerate(self.blocks):
                weight_size = block.weight.numel()
                if self.debug:
                    print(f"  Setting block {i} weights: {weight_size} elements at offset {offset}")
                block.weight.data.copy_(
                    weight_tensor[offset:offset + weight_size].view_as(block.weight)
                )
                offset += weight_size
        elif self.head_type == "turn":
            # FIXED: Only set linear layer weights (turn embeddings remain frozen)
            linear_weight_size = self.linear.weight.numel()
            if self.debug:
                print(f"  Setting linear weights: {linear_weight_size} elements at offset {offset}")
                print(f"  Turn embeddings remain frozen (not modified)")
            self.linear.weight.data.copy_(
                weight_tensor[offset:offset + linear_weight_size].view_as(self.linear.weight)
            )

        if self.debug:
            print(f"  Total elements processed: {offset}")

    def get_selected_dimensions(self) -> torch.Tensor:
        """Get the indices of currently selected dimensions (for sparse head only)."""
        if self.head_type != "sparse":
            raise ValueError(f"get_selected_dimensions() only available for sparse head, got {self.head_type}")

        sparsity_ratio = torch.sigmoid(self.sparsity_logit)
        target_k = max(1, int(self.hidden_size * (1 - sparsity_ratio)))
        _, top_indices = torch.topk(self.dimension_scores, target_k, dim=0)

        if self.debug:
            print(f"[RouterHead] Selected dimensions: {target_k}/{self.hidden_size}")
            print(f"  Sparsity ratio: {sparsity_ratio.item():.3f}")
            print(f"  Top indices (first 10): {top_indices[:10].tolist()}")

        return top_indices.sort()[0]  # Return sorted indices

    def get_adaptive_sparsity_stats(self) -> Dict[str, float]:
        """Get statistics about adaptive sparsity (for sparse head only)."""
        if self.head_type != "sparse":
            raise ValueError(f"get_adaptive_sparsity_stats() only available for sparse head, got {self.head_type}")

        sparsity_ratio = torch.sigmoid(self.sparsity_logit).item()
        target_k = max(1, int(self.hidden_size * (1 - sparsity_ratio)))
        temp = torch.clamp(self.selection_temperature, min=0.1, max=50.0).item()

        # Get current selection weights
        selection_weights = torch.softmax(self.dimension_scores / temp, dim=0)

        # For sparse heads, we don't have selection_threshold anymore
        # Use a simple threshold based on average weight
        threshold = selection_weights.mean().item()
        active_dims = (selection_weights > threshold).sum().item()

        stats = {
            "target_sparsity_ratio": sparsity_ratio,
            "target_k_dimensions": target_k,
            "actual_k_dimensions": int(target_k),
            "selection_temperature": temp,
            "active_dimensions": active_dims,
            "effective_sparsity": 1.0 - (active_dims / self.hidden_size),
            "score_std": self.dimension_scores.std().item(),
            "score_range": (self.dimension_scores.max() - self.dimension_scores.min()).item(),
        }

        if self.debug:
            print(f"[RouterHead] Adaptive sparsity stats:")
            for key, value in stats.items():
                print(f"  {key}: {value}")

        return stats

    def get_dimension_importance_stats(self) -> Dict[str, float]:
        """Get statistics about dimension importance (for sparse head only)."""
        if self.head_type != "sparse":
            raise ValueError(f"get_dimension_importance_stats() only available for sparse head, got {self.head_type}")

        sparsity_ratio = torch.sigmoid(self.sparsity_logit)
        target_k = max(1, int(self.hidden_size * (1 - sparsity_ratio)))

        scores = self.dimension_scores.detach().cpu()
        _, top_indices = torch.topk(scores, target_k, dim=0)

        unselected_mask = torch.ones(len(scores), dtype=torch.bool)
        unselected_mask[top_indices] = False

        stats = {
            "mean_selected_score": scores[top_indices].mean().item(),
            "mean_unselected_score": scores[unselected_mask].mean().item() if unselected_mask.sum() > 0 else 0.0,
            "score_std": scores.std().item(),
            "score_range": (scores.max() - scores.min()).item(),
            "selection_sparsity": target_k / self.hidden_size,
        }

        if self.debug:
            print(f"[RouterHead] Dimension importance stats:")
            for key, value in stats.items():
                print(f"  {key}: {value}")

        return stats


def create_router_head(
        hidden_size: int,
        num_agents: int,
        head_type: str = "linear",
        max_turns: int = 5,
        device: str = "cuda:0",
        dtype: torch.dtype = torch.bfloat16,
        debug: bool = False
) -> RouterHead:
    """Factory function to create router heads."""
    return RouterHead(hidden_size, num_agents, head_type, max_turns, device, dtype, debug)