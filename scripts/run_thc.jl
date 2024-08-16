using Oceananigans
using TopographicHorizontalConvection: HorizontalConvectionSimulation

#simulation = HorizontalConvectionSimulation(Ra=1e7, h₀_frac=0.6, Nx=256, Ny=1, Nz=32, advection=true)
#simulation = HorizontalConvectionSimulation(Ra=1e7, h₀_frac=0.6, Nx=256, Ny=1, Nz=32, advection=false)
#simulation = HorizontalConvectionSimulation(Ra=1e8, h₀_frac=0.6, Nx=256, Ny=1, Nz=32, advection=true)
simulation = HorizontalConvectionSimulation(Ra=1e8, h₀_frac=0.6, Nx=256, Ny=1, Nz=32, b_init=-0.5, advection=true)

run!(simulation, pickup=false)

