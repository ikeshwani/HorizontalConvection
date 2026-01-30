using Oceananigans
using TopographicHorizontalConvection: HorizontalConvectionSimulation

#simulation = HorizontalConvectionSimulation(Ra=1e7, h₀_frac=0.6, Nx=256, Ny=1, Nz=32, advection=true)
#simulation = HorizontalConvectionSimulation(Ra=1e7, h₀_frac=0.6, Nx=256, Ny=1, Nz=32, advection=false)
#simulation = HorizontalConvectionSimulation(Ra=1e8, h₀_frac=0.6, Nx=256, Ny=1, Nz=32, advection=true)
#simulation = HorizontalConvectionSimulation(Ra=1e5, h₀_frac=0.6, Nx=200, Ny=1, Nz=25, b_init=-0.5, coriolis = true, advection=true) #new res to resolve kolmogorov scale

simulation = HorizontalConvectionSimulation(;
   #domain parameters
    Nx = 1024, 
    Ny = 64,
    Nz = 128, 
    H = 1.0, 
    α = 8.0, 

    #topography parameters
    h₀_frac = 0.6, 
    numhill = 1, 

    #physics parameters
    Ra = 1e8,
    Pr = 1.0, 
    b★ = 1.0, 
    advection = true, 

    #initial conditions
    b_init = -0.5,

    #buoyancy forcing parameters
    winter_amplitude = 0.0,
    summer_amplitude = 0.0,
    seasonal_period = 0.0, 

    #coriolis parameters
    coriolis = false, 
    coriolis_scheme = :betaplane,
    latitude_south = -90.0, 
    latitude_north = 0.0, 
    coriolis_reference_latitude = nothing, #used for fplane only

    #wind forcing parameters
    wind = false, 
    wind_stress = 0.0, 
    wind_spatial_profile = nothing, 

    #southern westerlies shortcut
    use_SO_westerlies = false, 
    SO_westerlies_magnitude = 1e-4, 
    SO_westerlies_center_lat = -50.0,
    SO_westerlies_width = 10.0,

    #output parameters
    output_writer = true,
    output_dir = "../output/GPU_test/bss_Ra1e8",

    #computational parameters
    architecture = GPU()
)


run!(simulation, pickup=false)

