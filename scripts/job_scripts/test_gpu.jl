using CUDA
using Oceananigans
using TopographicHorizontalConvection:HC_sim_test_flat

println("CUDA Available: ", CUDA.functional())
println("CUDA Devices: ", length(devices()))

if CUDA.functional()
    println("GPU Device : ", name(device()))
    println("CUDA Version : ", CUDA.runtime_version())
end

test_sim = HC_sim_test_flat(; Nx =256, Ny=256, Nz=32)

run!(test_sim)