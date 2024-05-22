include("TopographicHorizontalConvection.jl")

#simulation = TopographicHorizontalConvection(h₀_frac=0.0, advection=true)
#simulation = TopographicHorizontalConvection(h₀_frac=0.0, advection=false)
simulation = TopographicHorizontalConvection(h₀_frac=0.6, advection=true)
#simulation = TopographicHorizontalConvection(h₀_frac=0.6, advection=false)
run!(simulation, pickup=true)

