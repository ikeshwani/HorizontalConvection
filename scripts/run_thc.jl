using TopographicHorizontalConvection: HorizontalConvectionSimulation

#simulation = HorizontalConvectionSimulation(h₀_frac=0.0, advection=true)
#simulation = HorizontalConvectionSimulation(h₀_frac=0.0, advection=false)
#simulation = HorizontalConvectionSimulation(h₀_frac=0.6, advection=true)
simulation = HorizontalConvectionSimulation(h₀_frac=0.6, advection=false)
run!(simulation, pickup=true)

