println(">>> starting run case 1 : 5x stretching <<<")

using CUDA
using Oceananigans
using TopographicHorizontalConvection:HorizontalConvectionSimulation

simulation = HorizontalConvectionSimulation(;
                   #domain parameters
    Nx = 512, 
    Ny = 512,
    Nz = 128, 
    H = 1.0, 
    α = 4.0,
    x_stretch = 0.12,   #5x stretch
    z_stretch = 2.56,   #5x stretch
    stop_time = 50.0,

    #topography parameters
    h₀_frac = 0.0, 
    numhill = 0, 

    #physics parameters
    Ra = 1e10,
    Pr = 1.0, 
    b★ = 1.0, 
    advection = true, 

    #initial conditions
    b_init = 0.0,

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
    output_dir = "/work/hdd/bfxn/ikeshwani/HorizontalConvection/output/GPU/chapter1_base/",

    #computational parameters
    architecture = GPU()
    )

run!(simulation, pickup=false)



