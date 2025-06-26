using Oceananigans
using TopographicHorizontalConvection: HorizontalConvectionSimulation
printf("Hello world!\n")


simulation = HorizontalConvectionSimulation(Ra=1e6, h₀_frac=0, Nx=951, Ny=1, Nz=119, b_init=-0.6, advection=true, architecture=CPU())
run!(simulation, pickup=false)