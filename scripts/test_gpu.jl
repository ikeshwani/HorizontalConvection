using CUDA
using Oceananigans

println("CUDA Available: ", CUDA.functional())
println("CUDA Devices: ", length(devices()))

if CUDA.functional()
    println("GPU Device : ", name(device()))
    println("CUDA Version : ", CUDA.runtime_version())
end

println("\n Testing simple grid creation ...")
grid = RectilinearGrid(GPU(), size=(10,10), x =(0,1), z=(0,1), topology=(Bounded, Flat, Bounded))
println("simple grid created successfully")

println("\n Test completed")