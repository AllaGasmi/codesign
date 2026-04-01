import pyopencl as cl
import numpy as np

# Taille de la matrice
N = 4096
TILE_SIZE = 16
WPT = 4  # Work Per Thread (register tiling)

# Initialisation des matrices
A = np.random.rand(N, N).astype(np.float32)
B = np.random.rand(N, N).astype(np.float32)
C = np.zeros((N, N), dtype=np.float32)

# Choisir le device NVIDIA
platform = cl.get_platforms()[0]  # NVIDIA
device = platform.get_devices()[0]  # MX550
ctx = cl.Context([device])
queue = cl.CommandQueue(ctx, 
        properties=cl.command_queue_properties.PROFILING_ENABLE)

kernel_code = """
#define TILE_SIZE 16
#define WPT 4

// ============================================
// KERNEL 1 : NAIVE
// ============================================
__kernel void matmul_naive(
    __global float* A,
    __global float* B,
    __global float* C,
    int N)
{
    int row = get_global_id(1);
    int col = get_global_id(0);
    
    float sum = 0.0f;
    for(int k = 0; k < N; k++) {
        sum += A[row*N + k] * B[k*N + col];
    }
    C[row*N + col] = sum;
}

// ============================================
// KERNEL 2 : COALESCED (reference)
// ============================================
__kernel void matmul_coalesced(
    __global float* A,
    __global float* B,
    __global float* C,
    int N)
{
    int col = get_global_id(0);
    int row = get_global_id(1);
    
    float sum = 0.0f;
    for(int k = 0; k < N; k++) {
        sum += A[row*N + k] * B[k*N + col];
    }
    C[row*N + col] = sum;
}

// ============================================
// KERNEL 3 : TILING (memoire locale)
// ============================================
__kernel void matmul_tiling(
    __global float* A,
    __global float* B,
    __global float* C,
    int N)
{
    __local float Asub[TILE_SIZE][TILE_SIZE];
    __local float Bsub[TILE_SIZE][TILE_SIZE];
    
    int row = get_global_id(1);
    int col = get_global_id(0);
    int localRow = get_local_id(1);
    int localCol = get_local_id(0);
    
    float sum = 0.0f;
    
    int numTiles = N / TILE_SIZE;
    for(int t = 0; t < numTiles; t++) {
        // Charger la tuile en memoire locale
        Asub[localRow][localCol] = A[row*N + t*TILE_SIZE + localCol];
        Bsub[localRow][localCol] = B[(t*TILE_SIZE + localRow)*N + col];
        
        barrier(CLK_LOCAL_MEM_FENCE);
        
        for(int k = 0; k < TILE_SIZE; k++) {
            sum += Asub[localRow][k] * Bsub[k][localCol];
        }
        
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    C[row*N + col] = sum;
}

// ============================================
// KERNEL 4 : REGISTER TILING (WPT)
// ============================================
__kernel void matmul_register(
    __global float* A,
    __global float* B,
    __global float* C,
    int N)
{
    __local float Asub[TILE_SIZE][TILE_SIZE];

    int row    = get_global_id(1);
    int col    = get_global_id(0) * WPT;
    int localRow = get_local_id(1);
    int localCol = get_local_id(0);

    // Registres prives pour WPT resultats
    float results[WPT];
    for(int w = 0; w < WPT; w++) {
        results[w] = 0.0f;
    }

    int numTiles = N / TILE_SIZE;
    for(int t = 0; t < numTiles; t++) {

        // Charger tuile A en memoire locale
        Asub[localRow][localCol] = A[row*N + t*TILE_SIZE + localCol];

        barrier(CLK_LOCAL_MEM_FENCE);

        // Chaque thread calcule WPT resultats
        for(int k = 0; k < TILE_SIZE; k++) {
            float a = Asub[localRow][k];
            for(int w = 0; w < WPT; w++) {
                results[w] += a * B[(t*TILE_SIZE + k)*N + col + w];
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Ecrire les WPT resultats dans C
    for(int w = 0; w < WPT; w++) {
        C[row*N + col + w] = results[w];
    }
}
"""

# Compiler
program = cl.Program(ctx, kernel_code).build()

# Buffers
mf = cl.mem_flags
A_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=A)
B_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=B)
C_buf = cl.Buffer(ctx, mf.WRITE_ONLY, C.nbytes)

# Work sizes
local_size      = (TILE_SIZE, TILE_SIZE)
global_size     = (N, N)
global_size_reg = (N//WPT, N)  # chaque thread fait WPT colonnes

# ============================================
# Fonction benchmark
# ============================================
def benchmark(kernel_func, global_s, local_s, label):
    # Warmup
    ev = kernel_func(queue, global_s, local_s,
                     A_buf, B_buf, C_buf, np.int32(N))
    ev.wait()
    # Mesure
    ev = kernel_func(queue, global_s, local_s,
                     A_buf, B_buf, C_buf, np.int32(N))
    ev.wait()
    t = (ev.profile.end - ev.profile.start) * 1e-9
    gflops = (2 * N**3) / (t * 1e9)
    print(f"{label:35s} : {t*1000:8.2f} ms  →  {gflops:7.2f} GFLOPS")
    return gflops

print("=" * 65)
print(f"  Matrix size: {N}x{N}    Work-group: {TILE_SIZE}x{TILE_SIZE}    WPT: {WPT}")
print("=" * 65)

# Benchmarks
gflops_naive  = benchmark(program.matmul_naive,
                           global_size, local_size,
                           "NAIVE")

gflops_coal   = benchmark(program.matmul_coalesced,
                           global_size, local_size,
                           "COALESCED (référence)")

gflops_tiling = benchmark(program.matmul_tiling,
                           global_size, local_size,
                           "TILING (mémoire locale)")

gflops_reg    = benchmark(program.matmul_register,
                           global_size_reg, local_size,
                           "REGISTER TILING (WPT=4)")

print("=" * 65)
print(f"\n📊 Gains par rapport au COALESCED :")
print(f"   TILING          : {gflops_tiling/gflops_coal:.2f}x")
print(f"   REGISTER TILING : {gflops_reg/gflops_coal:.2f}x")