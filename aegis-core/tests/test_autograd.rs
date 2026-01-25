use aegis_core::ml::tensor::Tensor;
use aegis_core::ml::autograd::{Variable, add, mul};

#[test]
fn test_autograd_simple() {
    // z = x * 2 + 5
    // x = 2
    // grad_x = 2
    
    let x_data = Tensor::new(&[2.0], &[1]);
    let x = Variable::new(x_data);
    
    let two_data = Tensor::new(&[2.0], &[1]);
    let two = Variable::new(two_data);
    
    let five_data = Tensor::new(&[5.0], &[1]);
    let five = Variable::new(five_data);
    
    let y = mul(&x, &two);
    let z = add(&y, &five);
    
    z.backward();
    
    let grad = x.grad().unwrap();
    assert_eq!(grad.get(&[0]), 2.0);
}

#[test]
fn test_autograd_matmul() {
    use aegis_core::ml::autograd::matmul;
    
    // C = A @ B
    // A: 1x2 [1, 2]
    // B: 2x1 [3, 4]
    // C: 1x1 [1*3 + 2*4] = [11]
    
    let a_data = Tensor::new(&[1.0, 2.0], &[1, 2]);
    let a = Variable::new(a_data);
    
    let b_data = Tensor::new(&[3.0, 4.0], &[2, 1]);
    let b = Variable::new(b_data);
    
    let c = matmul(&a, &b);
    c.backward();
    
    // dC = 1
    // dA = dC @ B^T = [1] @ [3, 4] = [3, 4]
    // dB = A^T @ dC = [1, 2]^T @ [1] = [1, 2]^T
    
    let grad_a = a.grad().unwrap();
    assert_eq!(grad_a.get(&[0, 0]), 3.0);
    assert_eq!(grad_a.get(&[0, 1]), 4.0);
    
    let grad_b = b.grad().unwrap();
    assert_eq!(grad_b.get(&[0, 0]), 1.0);
    assert_eq!(grad_b.get(&[1, 0]), 2.0);
}
