# 🧠 AEGIS Deep Learning Roadmap (PRD)

**Version:** 0.1.0-Draft
**Status:** Planning
**Target:** v0.2.0 "Synapse"

---

## 1. Executive Summary
AEGIS has established a "Geometric" foundation (Manifolds, TDA) and a "Pragmatic" layer (Candle/LLM integration). The next phase is to bridge these with a **Native Deep Learning Ecosystem** that rivals PyTorch in expressiveness but retains AEGIS's topological safety guarantees.

This PRD outlines the missing components required to transition AEGIS from "Experimental ML" to "Production DL".

## 2. Core Deficiencies (Gap Analysis)

| Feature | Current State | Required State | Priority |
|---------|---------------|----------------|----------|
| **Autograd** | Mocked/Simulated | Full DAG-based Reverse Mode Differentiation | 🔴 Critical |
| **Tensors** | `Vec<f32>` (CPU) | `wgpu`/`cuda` backed, N-dimensional, Strided | 🔴 Critical |
| **Optimizers** | Basic SGD | Adam, RMSProp, Adagrad with Weight Decay | 🟡 High |
| **Layers** | MLP, Conv2D | LSTM, GRU, TransformerBlock, Dropout, BatchNorm | 🟡 High |
| **Loss** | Implicit (in fit) | CrossEntropy, MSE, KL-Div as modular objects | 🟢 Medium |
| **Data** | Manual Lists | `DataLoader` with batching, shuffling, pre-fetch | 🟢 Medium |

## 3. Detailed Specifications

### 3.1 The Autograd Engine (`aegis-grad`)
We need a tape-based gradient engine that tracks the topological history of operations.

**Proposed Syntax:**
```aegis
let x = tensor([1.0, 2.0], requires_grad=true)
let y = x * 2 + 5
let z = y.mean()

z.backward() // Populates x.grad
```

**Implementation Strategy:**
-   Create `Variable` struct wrapping `Tensor` + `grad` + `creator`.
-   Implement `Function` trait with `forward` and `backward`.
-   Use topological sort on the computational graph (ironic, given our OS name) for backprop.

### 3.2 Hardware Acceleration (`aegis-compute`)
AEGIS must run on GPU to be taken seriously.

**Strategy:**
-   Leverage `wgpu` for cross-platform support (Metal/Vulkan/DX12).
-   Abstract the backend so the user just flags `device="gpu"`.
-   **Zero-Copy Manifolds:** Map GPU buffers directly to "Manifold" structs for real-time visualization without CPU roundtrip.

### 3.3 The "Synapse" Layer Library
Standardize the defining of neural architectures.

**Proposed Syntax:**
```aegis
class TransformerBlock {
    fn init(self, dim, heads) {
        self.attn = MultiHeadAttention(dim, heads)
        self.norm = LayerNorm(dim)
        self.ffn = FeedForward(dim)
    }

    fn forward(self, x) {
        let residual = x
        let x = self.attn(x)
        return self.norm(x + residual)
    }
}
```

### 3.4 Topological Loss Functions
Unique to AEGIS, we should implement loss functions that penalize topological complexity.

**New Concept: `PersistenceLoss`**
$$ L_{total} = L_{MSE} + \lambda \cdot L_{Betti} $$
*Penalize models that create fractured decision boundaries (high Betti numbers).*

## 4. Work Packages & Timeline

### Phase 1: Foundations (Week 1-2)
- [ ] Implement `Autograd` graph (Node/Edge structs).
- [ ] refactor `Tensor` to support `Rc<RefCell<Grad>>`.
- [ ] Port `SGD` to use the new Autograd.

### Phase 2: Acceleration (Week 3-4)
- [ ] Integrate `wgpu` compute shaders for `MatMul`.
- [ ] Implement `Device` switching (CPU/GPU).

### Phase 3: High-Level API (Week 5-6)
- [ ] Create `Optimizer` traits.
- [ ] Implement `Adam`.
- [ ] Build `Module` system (nestable layers).

## 5. Success Metrics
1.  **MNIST from Scratch:** Train a CNN to 99% accuracy using *only* native AEGIS Autograd (no Candle).
2.  **Transformer Training:** Train a generic NanoGPT on Shakespeare text.
3.  **Topological Regularization:** Demonstrate that `PersistenceLoss` reduces overfitting on noisy datasets compared to standard L2 regularization.
