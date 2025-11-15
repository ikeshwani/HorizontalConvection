using Oceananigans
using TopographicHorizontalConvection: HorizontalConvectionSimulation

#simulation = HorizontalConvectionSimulation(Ra=1e7, h₀_frac=0.6, Nx=256, Ny=1, Nz=32, advection=true)
#simulation = HorizontalConvectionSimulation(Ra=1e7, h₀_frac=0.6, Nx=256, Ny=1, Nz=32, advection=false)
#simulation = HorizontalConvectionSimulation(Ra=1e8, h₀_frac=0.6, Nx=256, Ny=1, Nz=32, advection=true)
#simulation = HorizontalConvectionSimulation(Ra=1e5, h₀_frac=0.6, numhill=1, Nx=169, Ny=1, Nz=21, b_init=-0.6, advection=true, architecture=CPU()) #new res to resolve kolmogorov scale
#simulation = HorizontalConvectionSimulation(Ra=5e6, h₀_frac=0.6, numhill=0, Nx=450, Ny=1, Nz=56, b_init=-0.6, advection=true, architecture=CPU()) #new res to resolve kolmogorov scale
#simulation = HorizontalConvectionSimulation(Ra=1e7, h₀_frac=0, numhill=0, Nx=535, Ny=1, Nz=67, b_init=-0.6, advection=true, architecture=CPU()) #new res to resolve kolmogorov scale


#simulation = HorizontalConvectionSimulation(Ra=1e6, h₀_frac=0.6, numhill = 1, Nx=1000, Ny=1, Nz=120, b_init=-0.6, advection=true, architecture=CPU())
simulation = HorizontalConvectionSimulation(Ra=1e5, h₀_frac=0.6, numhill = 1, Nx=50, Ny=1, Nz=6, b_init=-0.6, advection=true, architecture=CPU())


run!(simulation, pickup=false)
