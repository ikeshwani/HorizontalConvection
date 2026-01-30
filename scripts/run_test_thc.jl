println(">>> STARTING JULIA SCRIPT <<<")

println("julia version", VERSION)

# println("importing CUDA")
# using CUDA
# @info "CUDA functional?" CUDA.functional()
# CUDA.versioninfo()

println("import Oceananigans and Simulation Package")
using Oceananigans
using TopographicHorizontalConvection:HorizontalConvectionSimulation_TEST

println("packages imported successfully")

simulation = HorizontalConvectionSimulation_TEST(;
                   #domain parameters
    Nx = 256, 
    Ny = 16,
    Nz = 32, 
    H = 1.0, 
    α = 8.0,

    #topography parameters
    h₀_frac = 0.6, 
    numhill = 1, 

    #physics parameters
    Ra = 1e6,
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
    output_writer = false,
    output_dir = "/work/hdd/bfxn/ikeshwani/HorizontalConvection/output/GPU_test/test/Ra1e6",

    #computational parameters
    architecture = CPU()
    )

run!(simulation, pickup=false)

