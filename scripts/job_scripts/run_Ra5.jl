println("Running Ra1e5 4x stretching simulation from from t=1 to t=2, segment=2 and b_init=0")
#we change the stop time to 2.0 seconds, so that i can control when the sim time is over rather than


using CUDA
using Oceananigans
using TopographicHorizontalConvection:HorizontalConvectionSimulation

simulation = HorizontalConvectionSimulation(;
                   #domain parameters
    Nx = 512, 
    Ny = 256,
    Nz = 128, 
    H = 2.0, 
    α = 4.0,
    x_stretch = 4.0,   #4x stretch
    z_stretch = 4.0,   #4x stretch
    stop_time = 1.0,

    #topography parameters
    h₀_frac = 0.0, 
    numhill = 0, 

    #physics parameters
    Ra = 1e5,
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
    output_dir = "/work/hdd/bfxn/ikeshwani/HorizontalConvection/output/GPU/chapter1/RA1e5/4x_stretch/",
    segment = 2, 

    #computational parameters
    architecture = GPU()
    )

run!(simulation, pickup=false)

# save final checkpoint immediately after run completes
# @info "Run complete at t=$(time(simulation.model)), saving final checkpoint..."
# write_output!(simulation.output_writers[:checkpointer], simulation.model)
# @info "Final checkpoint saved."