/*
 * CUDA kernels for MeZO (Memory-efficient Zeroth-order) optimization
 * Optimized operations for SGLang integration
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <curand_kernel.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

#define BLOCK_SIZE 256
#define WARP_SIZE 32

// Error checking macro
#define CUDA_CHECK(call) do { \
    cudaError_t error = call; \
    if (error != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(error)); \
        exit(1); \
    } \
} while(0)

// Kernel 1: Fused perturbation generation and LoRA computation
template<typename scalar_t>
__global__ void mezo_fused_perturbation_lora_kernel(
    const scalar_t* __restrict__ A,        // LoRA A matrix (rank x hidden)
    const scalar_t* __restrict__ B,        // LoRA B matrix (hidden x rank)
    const scalar_t* __restrict__ W_base,   // Base weight matrix
    scalar_t* __restrict__ W_plus,         // Output: W_base + (B + eps*z_B) @ (A + eps*z_A)
    scalar_t* __restrict__ W_minus,        // Output: W_base + (B - eps*z_B) @ (A - eps*z_A)
    scalar_t* __restrict__ z_A,            // Storage for perturbation z_A
    scalar_t* __restrict__ z_B,            // Storage for perturbation z_B
    const float epsilon,
    const int hidden_dim,
    const int rank,
    curandState* states
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_elements_A = rank * hidden_dim;
    const int total_elements_B = hidden_dim * rank;
    
    // Initialize RNG state
    curandState local_state = states[tid];
    
    // Generate perturbations for A
    if (tid < total_elements_A) {
        float rand_val = curand_normal(&local_state);
        z_A[tid] = static_cast<scalar_t>(rand_val);
    }
    
    // Generate perturbations for B
    if (tid < total_elements_B) {
        float rand_val = curand_normal(&local_state);
        z_B[tid] = static_cast<scalar_t>(rand_val);
    }
    
    // Update RNG state
    states[tid] = local_state;
    
    __syncthreads();
    
    // Compute perturbed LoRA products
    // This is a simplified version - in practice, we'd use tensor cores
    const int row = tid / hidden_dim;
    const int col = tid % hidden_dim;
    
    if (row < hidden_dim && col < hidden_dim) {
        scalar_t sum_plus = 0;
        scalar_t sum_minus = 0;
        
        // Compute (B ± eps*z_B) @ (A ± eps*z_A)
        for (int k = 0; k < rank; k++) {
            scalar_t b_val = B[row * rank + k];
            scalar_t a_val = A[k * hidden_dim + col];
            scalar_t zb_val = z_B[row * rank + k];
            scalar_t za_val = z_A[k * hidden_dim + col];
            
            // Plus perturbation
            scalar_t b_plus = b_val + epsilon * zb_val;
            scalar_t a_plus = a_val + epsilon * za_val;
            sum_plus += b_plus * a_plus;
            
            // Minus perturbation
            scalar_t b_minus = b_val - epsilon * zb_val;
            scalar_t a_minus = a_val - epsilon * za_val;
            sum_minus += b_minus * a_minus;
        }
        
        // Add base weight
        const int idx = row * hidden_dim + col;
        W_plus[idx] = W_base[idx] + sum_plus;
        W_minus[idx] = W_base[idx] + sum_minus;
    }
}

// Kernel 2: Batched MeZO gradient accumulation
template<typename scalar_t>
__global__ void mezo_gradient_accumulation_kernel(
    const scalar_t* __restrict__ z_A,      // Perturbations for A
    const scalar_t* __restrict__ z_B,      // Perturbations for B
    const scalar_t* __restrict__ loss_diff, // (loss_plus - loss_minus) for each sample
    scalar_t* __restrict__ grad_A,         // Gradient accumulator for A
    scalar_t* __restrict__ grad_B,         // Gradient accumulator for B
    const float epsilon,
    const int n_samples,
    const int total_elements_A,
    const int total_elements_B
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Process gradient for A
    if (tid < total_elements_A) {
        scalar_t grad_sum = 0;
        for (int s = 0; s < n_samples; s++) {
            const int z_idx = s * total_elements_A + tid;
            scalar_t z_val = z_A[z_idx];
            scalar_t diff = loss_diff[s];
            grad_sum += z_val * diff / (2.0f * epsilon);
        }
        atomicAdd(&grad_A[tid], grad_sum / n_samples);
    }
    
    // Process gradient for B
    if (tid < total_elements_B) {
        scalar_t grad_sum = 0;
        for (int s = 0; s < n_samples; s++) {
            const int z_idx = s * total_elements_B + tid;
            scalar_t z_val = z_B[z_idx];
            scalar_t diff = loss_diff[s];
            grad_sum += z_val * diff / (2.0f * epsilon);
        }
        atomicAdd(&grad_B[tid], grad_sum / n_samples);
    }
}

// Kernel 3: Batched forward pass processing
template<typename scalar_t>
__global__ void mezo_batched_forward_kernel(
    const scalar_t* __restrict__ x,         // Input tensor
    const scalar_t* __restrict__ W_batch,   // Batched weight matrices
    scalar_t* __restrict__ y_batch,         // Output tensors
    const int batch_size,
    const int seq_len,
    const int hidden_dim,
    const int n_perturbations
) {
    // Simplified batched matrix multiplication
    // In practice, this would use tensor cores for efficiency
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_outputs = n_perturbations * batch_size * seq_len * hidden_dim;
    
    if (tid < total_outputs) {
        const int p = tid / (batch_size * seq_len * hidden_dim);
        const int b = (tid / (seq_len * hidden_dim)) % batch_size;
        const int s = (tid / hidden_dim) % seq_len;
        const int h = tid % hidden_dim;
        
        scalar_t sum = 0;
        for (int k = 0; k < hidden_dim; k++) {
            scalar_t x_val = x[b * seq_len * hidden_dim + s * hidden_dim + k];
            scalar_t w_val = W_batch[p * hidden_dim * hidden_dim + k * hidden_dim + h];
            sum += x_val * w_val;
        }
        
        y_batch[tid] = sum;
    }
}

// Helper function to initialize RNG states
__global__ void init_rng_states(curandState* states, unsigned long seed) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    curand_init(seed, tid, 0, &states[tid]);
}

// C++ interface functions
torch::Tensor mezo_fused_perturbation_lora(
    torch::Tensor A,
    torch::Tensor B,
    torch::Tensor W_base,
    float epsilon,
    int seed
) {
    // Check inputs
    TORCH_CHECK(A.is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "B must be a CUDA tensor");
    TORCH_CHECK(W_base.is_cuda(), "W_base must be a CUDA tensor");
    
    const int rank = A.size(0);
    const int hidden_dim = A.size(1);
    TORCH_CHECK(B.size(0) == hidden_dim);
    TORCH_CHECK(B.size(1) == rank);
    TORCH_CHECK(W_base.size(0) == hidden_dim);
    TORCH_CHECK(W_base.size(1) == hidden_dim);
    
    // Allocate output tensors
    auto options = A.options();
    auto W_plus = torch::empty_like(W_base);
    auto W_minus = torch::empty_like(W_base);
    auto z_A = torch::empty_like(A);
    auto z_B = torch::empty_like(B);
    
    // Allocate RNG states
    const int total_threads = (hidden_dim * hidden_dim + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE;
    curandState* d_states;
    CUDA_CHECK(cudaMalloc(&d_states, total_threads * sizeof(curandState)));
    
    // Initialize RNG
    init_rng_states<<<(total_threads + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
        d_states, seed
    );
    
    // Launch kernel
    const int blocks = (hidden_dim * hidden_dim + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(A.scalar_type(), "mezo_fused_perturbation_lora", ([&] {
        mezo_fused_perturbation_lora_kernel<scalar_t><<<blocks, BLOCK_SIZE>>>(
            A.data_ptr<scalar_t>(),
            B.data_ptr<scalar_t>(),
            W_base.data_ptr<scalar_t>(),
            W_plus.data_ptr<scalar_t>(),
            W_minus.data_ptr<scalar_t>(),
            z_A.data_ptr<scalar_t>(),
            z_B.data_ptr<scalar_t>(),
            epsilon,
            hidden_dim,
            rank,
            d_states
        );
    }));
    
    CUDA_CHECK(cudaFree(d_states));
    
    return torch::stack({W_plus, W_minus, z_A, z_B});
}

torch::Tensor mezo_gradient_accumulation(
    torch::Tensor z_A_batch,
    torch::Tensor z_B_batch,
    torch::Tensor loss_diff,
    float epsilon
) {
    // Check inputs
    TORCH_CHECK(z_A_batch.is_cuda(), "z_A_batch must be a CUDA tensor");
    TORCH_CHECK(z_B_batch.is_cuda(), "z_B_batch must be a CUDA tensor");
    TORCH_CHECK(loss_diff.is_cuda(), "loss_diff must be a CUDA tensor");
    
    const int n_samples = z_A_batch.size(0);
    const int rank = z_A_batch.size(1);
    const int hidden_dim = z_A_batch.size(2);
    
    // Allocate gradient accumulators
    auto options = z_A_batch.options();
    auto grad_A = torch::zeros({rank, hidden_dim}, options);
    auto grad_B = torch::zeros({hidden_dim, rank}, options);
    
    const int total_elements_A = rank * hidden_dim;
    const int total_elements_B = hidden_dim * rank;
    const int total_elements = std::max(total_elements_A, total_elements_B);
    const int blocks = (total_elements + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(z_A_batch.scalar_type(), "mezo_gradient_accumulation", ([&] {
        mezo_gradient_accumulation_kernel<scalar_t><<<blocks, BLOCK_SIZE>>>(
            z_A_batch.data_ptr<scalar_t>(),
            z_B_batch.data_ptr<scalar_t>(),
            loss_diff.data_ptr<scalar_t>(),
            grad_A.data_ptr<scalar_t>(),
            grad_B.data_ptr<scalar_t>(),
            epsilon,
            n_samples,
            total_elements_A,
            total_elements_B
        );
    }));
    
    return torch::stack({grad_A, grad_B});
}

// Python bindings
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_perturbation_lora", &mezo_fused_perturbation_lora, 
          "Fused MeZO perturbation and LoRA computation");
    m.def("gradient_accumulation", &mezo_gradient_accumulation,
          "MeZO gradient accumulation");
}