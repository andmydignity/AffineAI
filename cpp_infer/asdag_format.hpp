#pragma once

/**
 * ============================================================================
 * ASDAG Model Binary Specification (.asdag / .dagm)
 * ============================================================================
 * Magic: 'ASDA' (0x41534441)
 * Version: 2
 *
 * File Structure:
 * ┌────────────────────────────────────────────────────────────────┐
 * │ ASDAGHeader (64 bytes aligned)                                 │
 * ├────────────────────────────────────────────────────────────────┤
 * │ Tokenizer Metadata Block (JSON length + UTF-8 payload)          │
 * ├────────────────────────────────────────────────────────────────┤
 * │ Embedding Table: [vocab_size, d_model] float16 / float32       │
 * ├────────────────────────────────────────────────────────────────┤
 * │ Output Unembedding Table: [vocab_size, d_model] (optional tied)│
 * ├────────────────────────────────────────────────────────────────┤
 * │ Layers (0 .. n_layers - 1):                                    │
 * │   - Attention / RMSNorm Weights (Q, K, V, O, Norms)            │
 * │   - Router Hyperplane Weights: (2^tree_depth - 1, d_model) 2b  │
 * │   - Router Biases: (2^tree_depth - 1) float32                  │
 * │   - ASDAG 1:N Sparse Leaf Tables:                              │
 * │       * For each leaf (0 .. num_leaves - 1):                   │
 * │           - Leaf Bias: [d_model] float32                       │
 * │           - Norm Factor: float32                               │
 * │           - 1:N Sparse Compressed Index Table (nibbles/bytes)  │
 * │           - 1:N Sparse Non-Zero Sign Bit-Plane (1 bit/active)  │
 * └────────────────────────────────────────────────────────────────┘
 * ============================================================================
 */

#include <cstdint>

#pragma pack(push, 1)
struct ASDAGHeaderV2 {
    uint32_t magic = 0x41534441;        // 'ASDA'
    uint32_t version = 2;               // Specification v2
    uint32_t header_size = 128;         // Header size in bytes
    
    // Model Topology
    uint32_t vocab_size = 248320;
    uint32_t d_model = 2560;
    uint32_t n_layers = 32;
    uint32_t n_heads = 20;
    uint32_t n_kv_heads = 4;
    uint32_t d_head = 128;
    uint32_t max_seq_len = 32768;
    
    // ASDAG Polytope Routing & Sparsity
    uint32_t num_leaves = 16;
    uint32_t tree_depth = 4;            // 2^4 = 16 leaves
    uint32_t nm_n = 1;                  // Active non-zeros per group (1)
    uint32_t nm_m = 8;                  // Group size (8) -> 87.5% weight sparsity
    uint32_t shift_bits = 4;            // Shift4 (2^-p) activation precision
    uint32_t is_tied_embedding = 1;
    
    // Storage Alignments & Offsets
    uint64_t tokenizer_offset = 128;
    uint64_t tokenizer_size = 0;
    uint64_t embeddings_offset = 0;
    uint64_t layers_offset = 0;
    
    // Reserved padding to 128 bytes
    uint8_t reserved[48] = {0};
};
#pragma pack(pop)
