println(">>> starting run case comparison to bryden and nurser topography : for Ra1e6 4x stretching segment 20,stoptime=650 (90sec interval)<<<")

using CUDA
using Oceananigans
using TopographicHorizontalConvection:HorizontalConvectionSimulation

simulation = HorizontalConvectionSimulation(;
                   #domain parameters
    Nx = 512, 
    Ny = 32,
    Nz = 128, 
    H = 1.0, 
    α = 4.0,
    x_stretch = 4.0,   # 4x stretch
    z_stretch = 4.0,   # 4x stretch
    stop_time = 650.0,

    #topography parameters
    h₀_frac = 0.5, 
    numhill = 3, 

    #physics parameters
    Ra = 1e6,
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
    output_dir = "/work/hdd/bfxn/ikeshwani/HorizontalConvection/output/GPU/GRC/",
    segment = 20, 

    #computational parameters
    architecture = GPU()
    )

run!(simulation, pickup=true)

