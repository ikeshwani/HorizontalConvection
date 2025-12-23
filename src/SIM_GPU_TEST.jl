using Oceananigans
using CUDA
using Oceananigans.Solvers: ConjugateGradientPoissonSolver

Nx, Ny, Nz = 256, 16, 32
H = 1.0
Lx = 8*H
Ly = H/2
h₀_frac = 0.6
numhill = 2

if numhill == 1
    h₀_1 = h₀_frac * H
    h₀_2 = 0.0
elseif numhill == 2
    h₀_1 = h₀_frac * H 
    h₀_2 = h₀_frac * H
else
    h₀_1 = 0.0
    h₀_2 = 0.0
end  

hill_length = Lx / 32
channel_width = Ly / 8

# Seafloor function
function compute_seafloor_3d(x, y)
    hill_1 = (2/3) * h₀_1 * exp(-(x - 0.0 * Lx / 2)^2 / (2 * hill_length^2))
    hill_2 = h₀_2 * exp(-(x - 0.5 * Lx / 2)^2 / (2 * hill_length^2))
    channel = 1 - (1/3) * exp(-(y^2) / (2 * channel_width^2))
    return -H + (hill_1 + hill_2) * channel
end

arch = GPU()
println("Running on: ", arch)

cpu_grid = RectilinearGrid(
    CPU(), 
    size = (Nx, Ny, Nz), 
    x = (-Lx/2, Lx/2), 
    y = (-Ly/2, Ly/2), 
    z = (-H, 0), 
    halo = (4, 4, 4), 
    topology = (Bounded, Periodic, Bounded)
)

# Pre-compute seafloor heights on CPU
println("Computing seafloor heights on CPU...")
seafloor_heights = zeros(Nx, Ny)
for i in 1:Nx, j in 1:Ny
    x = cpu_grid.xᶜᵃᵃ[i]
    y = cpu_grid.yᵃᶜᵃ[j]
    seafloor_heights[i, j] = compute_seafloor_3d(x, y)
end

println("Seafloor height range: ", extrema(seafloor_heights))

# Convert to CuArray for GPU
seafloor_heights_gpu = CuArray(seafloor_heights)
println("Converted seafloor to GPU array")

# Create the actual grid on GPU
underlying_grid = RectilinearGrid(
    arch,
    size = (Nx, Ny, Nz),
    x = (-Lx/2, Lx/2),
    y = (-Ly/2, Ly/2),
    z = (-H, 0),
    halo = (4, 4, 4),
    topology = (Bounded, Periodic, Bounded)
)

println("Creating immersed boundary grid...")
grid = ImmersedBoundaryGrid(underlying_grid, GridFittedBottom(seafloor_heights_gpu))
println("✓ Grid created successfully!")

# Create simple model
println("Creating model...")

b★ = 1.0

# FIX: Pass Lx as a parameter instead of capturing it
@inline function surface_buoyancy_simple(x, y, t, p)
    bss = (tanh(3 * (x + p.Lx / 3)) - 1) / 2
    return p.b★ * (1 + bss)
end

# Create boundary condition with parameters
b_bcs = FieldBoundaryConditions(
    top = ValueBoundaryCondition(surface_buoyancy_simple, parameters=(Lx=Lx, b★=b★))
)

# Physics parameters
Ra = 1e11
Pr = 1.0
ν = sqrt(Pr * b★ * H^3 / Ra)
κ = ν / Pr

pressure_solver = ConjugateGradientPoissonSolver(grid)

model = NonhydrostaticModel(
    grid = grid,
    advection = WENO(),
    timestepper = :RungeKutta3,
    tracers = :b,
    buoyancy = BuoyancyTracer(),
    closure = ScalarDiffusivity(ν=ν, κ=κ),
    pressure_solver = pressure_solver,
    boundary_conditions = (b=b_bcs,)
)

println("✓ Model created successfully!")

# Set initial condition
println("Setting initial buoyancy...")
b_init = 0.0
@inline noise(x, z) = 1e-6 * sin(2π * x) * sin(2π * z)
bᵢ(x, y, z) = b_init + noise(x, z)
set!(model, b = bᵢ)

println("✓ Initial conditions set!")

# Create simulation
τ_eq = sqrt(Ra)
min_Δz = minimum_zspacing(grid)
diffusive_time_scale = min_Δz^2 / κ
advective_time_scale = sqrt(min_Δz / b★)
Δt = 0.1 * minimum([diffusive_time_scale, advective_time_scale])

simulation = Simulation(model, Δt=Δt, stop_time=10.0)

println("✓ Simulation created!")
println("Δt = ", Δt)
println("Stop time = ", simulation.stop_time)

println("\n" * "="^60)
println("SUCCESS! Everything is working on GPU!")
println("="^60)
println("\nRunning simulation...")

run!(simulation)

println("Simulation completed!")