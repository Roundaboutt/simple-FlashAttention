#include <torch/types.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ 
void forward_kernel(const float* Q, const float* K, const float* V, const int N, const int d,
                    const int Tc, const int Tr, const int Bc, const int Br, const float softmax_scale,
                    float* l, float* m, float* O) {

    // 每个 thread 负责一行(Bc)
    const int tid = threadIdx.x;

    const int bx = blockIdx.x;  // batch index
    const int by = blockIdx.y;  // head index

    extern __shared__ float sram[];
    const int tile_size_kv = Bc * d;
    const int tile_size_q = Br * d;

    float* Qi = sram;                           // size: Br*d
    float* Kj = &sram[Br * d];                  // size: Bc*d
    float* Vj = &sram[Br * d + Bc * d];         // size: Bc*d
    float* S = &sram[Br * d + Bc * d * 2];      // size: Br*Bc  缓存 QK^T 的结果
    
    //                     跳过前面的batch              跳过当前batch的head
    const int qkv_offset = (bx * gridDim.y * N * d) + (by * N * d); // gridDim.y = nh
    const int lm_offset = (bx * gridDim.y * N) + (by * N);  // l, m少了d维度
    
    // 当前线程要访问 qkv 的位置
    const float* Q_start = Q + qkv_offset;
    const float* K_start = K + qkv_offset;
    const float* V_start = V + qkv_offset;
    float* O_start = O + qkv_offset;

    float* l_start = l + lm_offset;
    float* m_start = m + lm_offset;

    // outer loop
    for (int j = 0; j < Tc; j++){
        // 把 K, V搬到sram
        for (int x = 0; x < d; x++){

            // Bc != Br 时,这个判断逻辑必须要加
            if (tid < Bc){
                Kj[tid * d + x] = K_start[tile_size_kv * j + tid * d + x];
                Vj[tid * d + x] = V_start[tile_size_kv * j + tid * d + x];                
            }

        } 

        __syncthreads();      // 如果Br = Bc = 32, 则可以不用同步,因为恰好是一个warp

        // inner loop
        for (int i = 0; i < Tr; i++){
            // 把 Q 搬到sram
            for (int x = 0; x < d; x++){
                Qi[tid * d + x] = Q_start[tile_size_q * i + tid * d + x];
            }

            // 每行都有一个 m 和 l, 
            float row_m_prev = m_start[Br * i + tid];
            float row_l_prev = l_start[Br * i + tid];

            float row_m = -INFINITY;
            // 计算S (Br x Bc) 一个线程只计算一行
            for (int y = 0; y < Bc; y++){
                float sum = 0.f;
                for (int x = 0; x < d; x++){
                    sum += Qi[tid * d + x] * Kj[y * d + x];
                }
                sum *= softmax_scale;
                S[tid * Bc + y] = sum;
                row_m = fmaxf(row_m, sum);
            }

            float row_l = 0;
            // 计算每一行的和
            for (int x = 0; x < Bc; x++){
                S[Bc * tid + x] = __expf(S[Bc * tid + x] - row_m);
                row_l += S[Bc * tid + x];
            }

            // 计算新的 m 和 l
            float row_m_new = fmaxf(row_m_prev, row_m);
            float row_l_new = (__expf(row_m_prev - row_m_new) * row_l_prev) + (__expf(row_m - row_m_new) * row_l);
            
            for (int x = 0; x < d; x++) {
                float pv = 0;  // Pij * Vj
                for (int y = 0; y < Bc; y++) {
                    pv += S[(Bc * tid) + y] * Vj[(y * d) + x];
                }
                O_start[(tile_size_q * i) + (tid * d) + x] = (1 / row_l_new) \
                    * ((row_l_prev * __expf(row_m_prev - row_m_new) * O_start[(tile_size_q * i) + (tid * d) + x]) \
                    + (__expf(row_m - row_m_new) * pv));
            }
            
            m_start[Br * i + tid] = row_m_new;
            l_start[Br * i + tid] = row_l_new;
        }
        __syncthreads();
    }
}

torch::Tensor forward(torch::Tensor Q, torch::Tensor K, torch::Tensor V) {
    // 这里 Bc 必须小于 Br
    const int Br = 64; const int Bc = 32;
    const int batch = Q.size(0);
    const int N_heads = Q.size(1);
    const int N = Q.size(2);
    const int d = Q.size(3);

    const int Tr = ceil(N / Br); const int Tc = ceil(N / Bc);
    const float softmax_scale = rsqrtf(d);

    auto O = torch::zeros_like(Q);
    auto l = torch::zeros({batch, N_heads, N});
    auto m = torch::full({batch, N_heads, N}, -INFINITY);

    torch::Device device(torch::kCUDA);
    l = l.to(device);
    m = m.to(device);

    const int sram_size = (2 * Br * d * sizeof(float)) + (Bc * d * sizeof(float)) + (Bc * Br * sizeof(float));

    dim3 gridSize(batch, N_heads);
    dim3 blockSize(Br);     // 每个线程负责一行, 一共有Br行, 所以一个block就有Br个线程

    int max_sram_size;
    cudaDeviceGetAttribute(&max_sram_size, cudaDevAttrMaxSharedMemoryPerBlock, 0);
    printf("Max shared memory: %d, requested shared memory: %d \n", max_sram_size, sram_size);

    forward_kernel<<<gridSize, blockSize, sram_size>>>(
        Q.data_ptr<float>(), K.data_ptr<float>(), V.data_ptr<float>(),
        N, d, Tc, Tr, Bc, Br, softmax_scale,
        l.data_ptr<float>(),
        m.data_ptr<float>(),
        O.data_ptr<float>()
    );
    return O;
}