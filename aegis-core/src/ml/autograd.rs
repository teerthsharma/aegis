//! ═══════════════════════════════════════════════════════════════════════════════
//! AEGIS Autograd Engine
//! ═══════════════════════════════════════════════════════════════════════════════
//!
//! Reverse-mode automatic differentiation graph.
//!
//! ═══════════════════════════════════════════════════════════════════════════════

use alloc::rc::Rc;
use alloc::vec::Vec;
use alloc::vec;
use core::cell::RefCell;
use crate::ml::tensor::Tensor;

/// A node in the computation graph
#[derive(Clone)]
pub struct Variable {
    pub data: Tensor,
    pub grad: Rc<RefCell<Option<Tensor>>>,
    pub creator: Option<Rc<dyn Function>>,
}

impl Variable {
    /// Create a new leaf variable
    pub fn new(data: Tensor) -> Self {
        Self {
            data,
            grad: Rc::new(RefCell::new(None)),
            creator: None,
        }
    }

    /// Access gradient
    pub fn grad(&self) -> Option<Tensor> {
        self.grad.borrow().clone()
    }

    /// Zero out gradient
    pub fn zero_grad(&self) {
        *self.grad.borrow_mut() = None;
    }

    /// Backpropagate gradients
    pub fn backward(&self) {
        // Topological sort not strictly needed if we just recurse, 
        // but for a proper engine we want BFS/DFS or topo sort.
        // For this MVP, we'll use simple recursion with implicit stack.
        // In a real deep graph, this might overflow, so a queue is better.
        
        // Seed gradient with 1.0s if empty
        if self.grad.borrow().is_none() {
            let ones = Tensor::ones(&self.data.shape);
            *self.grad.borrow_mut() = Some(ones);
        }

        // We need to traverse the graph. 
        // Simplest valid approach for Phase 1: Recursive DFS
        if let Some(ref func) = self.creator {
            let grads = func.backward(self.grad.borrow().as_ref().unwrap());
            let inputs = func.inputs();
            
            for (i, input) in inputs.iter().enumerate() {
                let mut current_grad = input.grad.borrow_mut();
                if let Some(ref mut g) = *current_grad {
                    *g = g.add(&grads[i]);
                } else {
                    *current_grad = Some(grads[i].clone());
                }
                drop(current_grad); // Release borrow
                
                // Recurse
                input.backward(); 
            }
        }
    }
}

/// Differentiable Function trait
pub trait Function {
    fn forward(&self) -> Tensor;
    fn backward(&self, grad_output: &Tensor) -> Vec<Tensor>;
    fn inputs(&self) -> Vec<Variable>;
}

// ═══════════════════════════════════════════════════════════════════════════════
// Operations
// ═══════════════════════════════════════════════════════════════════════════════

// --- Add ---
pub struct Add {
    a: Variable,
    b: Variable,
}

impl Function for Add {
    fn forward(&self) -> Tensor {
        self.a.data.add(&self.b.data)
    }

    fn backward(&self, grad_output: &Tensor) -> Vec<Tensor> {
        // d/dx (x+y) = 1 * grad_output
        // d/dy (x+y) = 1 * grad_output
        vec![grad_output.clone(), grad_output.clone()]
    }

    fn inputs(&self) -> Vec<Variable> {
        vec![self.a.clone(), self.b.clone()]
    }
}

pub fn add(a: &Variable, b: &Variable) -> Variable {
    let func = Rc::new(Add { a: a.clone(), b: b.clone() });
    let data = func.forward();
    Variable {
        data,
        grad: Rc::new(RefCell::new(None)),
        creator: Some(func),
    }
}

// --- Mul ---
pub struct Mul {
    a: Variable,
    b: Variable,
}

impl Function for Mul {
    fn forward(&self) -> Tensor {
        self.a.data.mul(&self.b.data)
    }

    fn backward(&self, grad_output: &Tensor) -> Vec<Tensor> {
        // d/da (a*b) = b * grad_output
        // d/db (a*b) = a * grad_output
        let da = grad_output.mul(&self.b.data);
        let db = grad_output.mul(&self.a.data);
        vec![da, db]
    }

    fn inputs(&self) -> Vec<Variable> {
        vec![self.a.clone(), self.b.clone()]
    }
}

pub fn mul(a: &Variable, b: &Variable) -> Variable {
    let func = Rc::new(Mul { a: a.clone(), b: b.clone() });
    let data = func.forward();
    Variable {
        data,
        grad: Rc::new(RefCell::new(None)),
        creator: Some(func),
    }
}

// --- MatMul ---
pub struct MatMul {
    a: Variable,
    b: Variable,
}

impl Function for MatMul {
    fn forward(&self) -> Tensor {
        self.a.data.matmul(&self.b.data)
    }

    fn backward(&self, grad_output: &Tensor) -> Vec<Tensor> {
        // C = A @ B
        // dA = dC @ B^T
        // dB = A^T @ dC
        let da = grad_output.matmul(&self.b.data.transpose());
        let db = self.a.data.transpose().matmul(grad_output);
        vec![da, db]
    }

    fn inputs(&self) -> Vec<Variable> {
        vec![self.a.clone(), self.b.clone()]
    }
}

pub fn matmul(a: &Variable, b: &Variable) -> Variable {
    let func = Rc::new(MatMul { a: a.clone(), b: b.clone() });
    let data = func.forward();
    Variable {
        data,
        grad: Rc::new(RefCell::new(None)),
        creator: Some(func),
    }
}

// --- ReLU ---
pub struct ReLU {
    input: Variable,
}

impl Function for ReLU {
    fn forward(&self) -> Tensor {
        let total_size: usize = self.input.data.shape.iter().product();
        let mut result_data = Vec::with_capacity(total_size);
        let data = self.input.data.data.borrow();
        
        for &val in data.iter() {
            result_data.push(if val > 0.0 { val } else { 0.0 });
        }
        
        Tensor::new(&result_data, &self.input.data.shape)
    }

    fn backward(&self, grad_output: &Tensor) -> Vec<Tensor> {
        let total_size: usize = self.input.data.shape.iter().product();
        let mut grad_data = Vec::with_capacity(total_size);
        let input_data = self.input.data.data.borrow();
        let grad_out_data = grad_output.data.borrow();
        
        for i in 0..total_size {
            grad_data.push(if input_data[i] > 0.0 { grad_out_data[i] } else { 0.0 });
        }
        
        vec![Tensor::new(&grad_data, &self.input.data.shape)]
    }

    fn inputs(&self) -> Vec<Variable> {
        vec![self.input.clone()]
    }
}

pub fn relu(a: &Variable) -> Variable {
    let func = Rc::new(ReLU { input: a.clone() });
    let data = func.forward();
    Variable {
        data,
        grad: Rc::new(RefCell::new(None)),
        creator: Some(func),
    }
}
