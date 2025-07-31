#![no_std]
#![no_main]
#![feature(asm_const)]

extern crate alloc;
use alloc::vec::Vec;
use alloc::format;

// allocate memory for stack
use polkavm_derive::min_stack_size;
min_stack_size!(65536); // 64KB stack

// allocate memory for heap
use simplealloc::SimpleAlloc;
#[global_allocator]
static ALLOCATOR: SimpleAlloc<524288> = SimpleAlloc::new(); // 512KB heap

use utils::functions::call_log;

const INPUT_SIZE: usize = 784;
const HIDDEN_SIZE: usize = 128;
const OUTPUT_SIZE: usize = 10;

struct QuantParams {
    weight_scale: f32,
    weight_zero_point: u8,
    bias_scale: f32,
    bias_zero_point: u8,
}

struct NeuralNetwork {
    fc1_weights: Vec<u8>,
    fc1_bias: Vec<u8>,
    fc2_weights: Vec<u8>,
    fc2_bias: Vec<u8>,
    fc1_params: QuantParams,
    fc2_params: QuantParams,
}

impl NeuralNetwork {
    fn new() -> Self {
        NeuralNetwork {
            fc1_weights: Vec::new(),
            fc1_bias: Vec::new(),
            fc2_weights: Vec::new(),
            fc2_bias: Vec::new(),
            fc1_params: QuantParams {
                weight_scale: 0.003957126755267382,
                weight_zero_point: 161,
                bias_scale: 0.0003837844415102154,
                bias_zero_point: 143,
            },
            fc2_params: QuantParams {
                weight_scale: 0.003803275991231203,
                weight_zero_point: 161,
                bias_scale: 0.0005365432007238269,
                bias_zero_point: 190,
            },
        }
    }

    fn load_weights(&mut self) -> Result<(), &'static str> {
        call_log(2, None, &format!("Loading weights in PVM environment using include_bytes!"));
        
        // Load real weights using include_bytes!
        // Using the correct paths from weights/ directory
        const FC1_WEIGHTS_DATA: &[u8] = include_bytes!("../weights/fc1_weight.bin");
        const FC1_BIAS_DATA: &[u8] = include_bytes!("../weights/fc1_bias.bin");
        const FC2_WEIGHTS_DATA: &[u8] = include_bytes!("../weights/fc2_weight.bin");
        const FC2_BIAS_DATA: &[u8] = include_bytes!("../weights/fc2_bias.bin");
        
        // Verify the sizes match expected dimensions
        if FC1_WEIGHTS_DATA.len() != INPUT_SIZE * HIDDEN_SIZE {
            call_log(2, None, &format!("❌ FC1 weights size mismatch: expected {}, got {}", 
                INPUT_SIZE * HIDDEN_SIZE, FC1_WEIGHTS_DATA.len()));
            return Err("FC1 weights size mismatch");
        }
        
        if FC1_BIAS_DATA.len() != HIDDEN_SIZE {
            call_log(2, None, &format!("❌ FC1 bias size mismatch: expected {}, got {}", 
                HIDDEN_SIZE, FC1_BIAS_DATA.len()));
            return Err("FC1 bias size mismatch");
        }
        
        if FC2_WEIGHTS_DATA.len() != HIDDEN_SIZE * OUTPUT_SIZE {
            call_log(2, None, &format!("❌ FC2 weights size mismatch: expected {}, got {}", 
                HIDDEN_SIZE * OUTPUT_SIZE, FC2_WEIGHTS_DATA.len()));
            return Err("FC2 weights size mismatch");
        }
        
        if FC2_BIAS_DATA.len() != OUTPUT_SIZE {
            call_log(2, None, &format!("❌ FC2 bias size mismatch: expected {}, got {}", 
                OUTPUT_SIZE, FC2_BIAS_DATA.len()));
            return Err("FC2 bias size mismatch");
        }
        
        // Copy the data to our vectors
        self.fc1_weights = FC1_WEIGHTS_DATA.to_vec();
        self.fc1_bias = FC1_BIAS_DATA.to_vec();
        self.fc2_weights = FC2_WEIGHTS_DATA.to_vec();
        self.fc2_bias = FC2_BIAS_DATA.to_vec();

        call_log(2, None, &format!("✅ Real weights loaded successfully"));
        call_log(2, None, &format!("   FC1 weights: {} elements", self.fc1_weights.len()));
        call_log(2, None, &format!("   FC1 bias: {} elements", self.fc1_bias.len()));
        call_log(2, None, &format!("   FC2 weights: {} elements", self.fc2_weights.len()));
        call_log(2, None, &format!("   FC2 bias: {} elements", self.fc2_bias.len()));

        Ok(())
    }

    fn dequantize_weight(&self, quantized: u8, params: &QuantParams) -> f32 {
        params.weight_scale * (quantized as f32 - params.weight_zero_point as f32)
    }

    fn dequantize_bias(&self, quantized: u8, params: &QuantParams) -> f32 {
        params.bias_scale * (quantized as f32 - params.bias_zero_point as f32)
    }

    fn quantize_input(&self, input: &[u8]) -> Vec<f32> {
        // Normalize input to [-1, 1] range (same as during training)
        input.iter().map(|&x| {
            (x as f32 / 255.0) * 2.0 - 1.0
        }).collect()
    }

    fn argmax(&self, values: &[f32]) -> (usize, f32) {
        let (idx, &val) = values.iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap_or((0, &0.0));
        (idx, val)
    }

    fn relu(x: f32) -> f32 {
        if x > 0.0 { x } else { 0.0 }
    }

    fn infer_float(&self, input: &[u8]) -> usize {
        call_log(2, None, &format!("\n🔍 Detailed inference process (float version):"));
        
        // Normalize input
        let normalized_input = self.quantize_input(input);
        let (mut input_min, mut input_max, mut input_sum) = (normalized_input[0], normalized_input[0], 0.0f32);
        for &val in &normalized_input {
            if val < input_min { input_min = val; }
            if val > input_max { input_max = val; }
            input_sum += val;
        }
        let input_avg = input_sum / normalized_input.len() as f32;
        call_log(2, None, &format!("   Normalized input range: {:.4} - {:.4}, mean: {:.4}", input_min, input_max, input_avg));

        // First layer calculation
        let mut fc1_output = Vec::with_capacity(HIDDEN_SIZE);
        for row in 0..HIDDEN_SIZE {
            let mut sum = 0.0f32;
            for col in 0..INPUT_SIZE {
                let weight_idx = row * INPUT_SIZE + col;
                let weight = self.dequantize_weight(self.fc1_weights[weight_idx], &self.fc1_params);
                sum += weight * normalized_input[col];
            }
            // Add bias
            let bias = self.dequantize_bias(self.fc1_bias[row], &self.fc1_params);
            sum += bias;
            fc1_output.push(sum);
        }

        let (mut fc1_min, mut fc1_max, mut fc1_sum) = (fc1_output[0], fc1_output[0], 0.0f32);
        for &val in &fc1_output {
            if val < fc1_min { fc1_min = val; }
            if val > fc1_max { fc1_max = val; }
            fc1_sum += val;
        }
        let fc1_avg = fc1_sum / fc1_output.len() as f32;
        call_log(2, None, &format!("   FC1 output range: {:.4} - {:.4}, mean: {:.4}", fc1_min, fc1_max, fc1_avg));

        // ReLU activation
        let fc1_activated: Vec<f32> = fc1_output.iter().map(|&x| Self::relu(x)).collect();

        let (mut relu_min, mut relu_max, mut relu_sum) = (fc1_activated[0], fc1_activated[0], 0.0f32);
        for &val in &fc1_activated {
            if val < relu_min { relu_min = val; }
            if val > relu_max { relu_max = val; }
            relu_sum += val;
        }
        let relu_avg = relu_sum / fc1_activated.len() as f32;
        call_log(2, None, &format!("   ReLU output range: {:.4} - {:.4}, mean: {:.4}", relu_min, relu_max, relu_avg));

        // Second layer calculation
        let mut fc2_output = Vec::with_capacity(OUTPUT_SIZE);
        for row in 0..OUTPUT_SIZE {
            let mut sum = 0.0f32;
            for col in 0..HIDDEN_SIZE {
                let weight_idx = row * HIDDEN_SIZE + col;
                let weight = self.dequantize_weight(self.fc2_weights[weight_idx], &self.fc2_params);
                sum += weight * fc1_activated[col];
            }
            let bias = self.dequantize_bias(self.fc2_bias[row], &self.fc2_params);
            sum += bias;
            fc2_output.push(sum);
        }

        let (mut fc2_min, mut fc2_max) = (fc2_output[0], fc2_output[0]);
        for &val in &fc2_output {
            if val < fc2_min { fc2_min = val; }
            if val > fc2_max { fc2_max = val; }
        }
        call_log(2, None, &format!("   FC2 output range: {:.4} - {:.4}", fc2_min, fc2_max));

        // Show all output values
        call_log(2, None, &format!("   FC2 class scores:"));
        for (i, &score) in fc2_output.iter().enumerate() {
            call_log(2, None, &format!("     Class {}: {:.4}", i, score));
        }

        let (prediction, max_score) = self.argmax(&fc2_output);
        call_log(2, None, &format!("   Final prediction: class {} (score: {:.4})", prediction, max_score));

        prediction
    }
}

#[polkavm_derive::polkavm_export]
extern "C" fn refine(_start_address: u64, _length: u64) -> (u64, u64) {
    call_log(2, None, &format!("🧠 Rust inference engine - PVM version"));
    call_log(2, None, &format!("================================"));

    let mut nn = NeuralNetwork::new();
    if let Err(e) = nn.load_weights() {
        call_log(2, None, &format!("❌ Failed to load weights: {}", e));
        return (0, 1); // Return error code
    }

    // Load real test image using include_bytes!
    // Using the correct path from test_data/ directory
    const TEST_IMAGE_DATA: &[u8] = include_bytes!("../test_data/test_image.bin");
    
    if TEST_IMAGE_DATA.len() != INPUT_SIZE {
        call_log(2, None, &format!("❌ Test image size mismatch: expected {}, got {}", 
            INPUT_SIZE, TEST_IMAGE_DATA.len()));
        return (0, 1); // Return error code
    }
    
    call_log(2, None, &format!("\n🖼️  Test image info:"));
    let img_stats = (
        *TEST_IMAGE_DATA.iter().min().unwrap(),
        *TEST_IMAGE_DATA.iter().max().unwrap(),
        TEST_IMAGE_DATA.iter().map(|&x| x as f32).sum::<f32>() / TEST_IMAGE_DATA.len() as f32
    );
    call_log(2, None, &format!("   Raw image range: {} - {}, mean: {:.2}", img_stats.0, img_stats.1, img_stats.2));

    let prediction = nn.infer_float(TEST_IMAGE_DATA);

    call_log(2, None, &format!("\n📋 Final result: predicted digit = {}", prediction));

    (prediction as u64, 0) // Return prediction and success code
}

#[polkavm_derive::polkavm_export]
extern "C" fn accumulate(_start_address: u64, _length: u64) -> (u64, u64) {
    call_log(2, None, &format!("MNIST accumulate function called"));
    (0, 0)
}

#[polkavm_derive::polkavm_export]
extern "C" fn on_transfer(_start_address: u64, _length: u64) -> (u64, u64) {
    call_log(2, None, &format!("MNIST on_transfer function called"));
    (0, 0)
}
