println(">>> starting run case 1 : 2x stretching <<<")

using CUDA
using Oceananigans
using TopographicHorizontalConvection:HorizontalConvectionSimulation

simulation = HorizontalConvectionSimulation(;
                   #domain parameters
    Nx = 512, 
    Ny = 32,
    Nz = 64, 
    H = 1.0, 
    α = 8.0,
    x_stretch = 0.24,   #2x stretch
    z_stretch = 0.41,   #2x stretch
    stop_time = 10.0,

    #topography parameters
    h₀_frac = 0.6, 
    numhill = 1, 

    #physics parameters
    Ra = 1e9,
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
    output_dir = "/work/hdd/bfxn/ikeshwani/HorizontalConvection/output/GPU_test/cheb_2x_stretch/b_base/Ra1e9/",

    #computational parameters
    architecture = GPU()
    )

run!(simulation, pickup=false)

