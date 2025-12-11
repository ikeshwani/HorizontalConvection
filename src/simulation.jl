# # Hilly horizontal convection
#
# In "horizontal convection", a non-uniform buoyancy is imposed on top of an initially resting fluid.
# This script is modified from the horizontal convection example in the Oceananigans documentation.
# We modify the structure of the surface boundary condition, the model parameters, and the bottom topography.
#
# ## Horizontal convection
#
# We consider two-dimensional horizontal convection of an incompressible flow ``\boldsymbol{u} = (u, w)``
# on the ``(x, z)``-plane (``-L_x/2 \le x \le L_x/2`` and ``-H \le z \le 0``). The flow evolves
# under the effect of gravity. The only forcing on the fluid comes from a prescribed, non-uniform
# buoyancy at the top-surface of the domain.

#physical constants 
const Ω = 7.292115e-5 # earth's rotation rate in s⁻¹
const R = 6.371e6 # earth's radius in m

using Oceananigans
using Printf
using Oceananigans.AbstractOperations: KernelFunctionOperation
using Oceananigans.Solvers: ConjugateGradientPoissonSolver
using Oceananigans.Grids: Center
using Oceananigans.BuoyancyFormulations: Zᶜᶜᶜ
using Oceanostics

# helper functions for potential energy

@inline function PotentialEnergy(model)
    
    b = model.tracers.b
    grid = model.grid
    return KernelFunctionOperation{Center, Center, Center}(bz_ccc, grid, b)
end

@inline bz_ccc(i, j, k, grid, b) = - b[i, j, k] * Zᶜᶜᶜ(i, j, k, grid)

# ----------------------------------------------------------------------- #

# Domain Constructor 
"""
using a struct and function for configuring the domain 
    struct : defines the "type" of variables
    the function: constructs the variables 
"""

struct DomainConfig
    H::Float64
    Lx::Float64
    Ly::Float64
    Nx::Int
    Ny::Int
    Nz::Int
end

function DomainConfig(; H= 1.0, α = 8.0, Nx = 256, Ny = 1, Nz = 32)
    Lx = α * H
    Ly = H/4
    return DomainConfig(H, Lx, Ly, Nx, Ny, Nz)
end

# construct the seafloor

function make_seafloor(domain::DomainConfig, h₀_frac, numhill)
    """
    make_seafloor :
    uses the domain struct : DomainConfig which contains H, Lx, Ly
    h₀_frac : fraction of domain height for hill amplitude
    numhill : Number of hills (0,1, or 2)

    returns seafloor function to input into ImmersedBoundaryGrid
    """
    
    H, Lx, Ly = domain.H, domain.Lx, domain.Ly


    if numhill == 1
        h₀_1 = h₀_frac * H
        h₀_2 = 0.0
    elseif numhill == 2
        h₀_1 = h₀_frac * H 
        h₀_2 = h₀_frac * H
    elseif numhill == 0 
        h₀_1 = 0.0
        h₀_2 = 0.0
    end

    hill_length = Lx/32

    #define individual hills
    hill_1(x) = (2/3) * h₀_1 * exp(-(x - 0.0Lx/2)^2 / (2*hill_length^2))
    hill_2(x) = h₀_2 * exp(-(x-0.5Lx/2)^2 / (2*hill_length^2))

    if domain.Ny == 1
        # 2D case !
        seafloor_flaty(x) = -H + (hill_1(x) + hill_2(x))
        return seafloor_flaty
    else
        # 3D case : add meridional channel
        channel_width = Ly/8
        channel(y) = 1 - (1/3) * exp(-(y^2) / (2 * channel_width^2))
        seafloor(x,y) = -H + (hill_1(x) + hill_2(x)) * channel(y)
        return seafloor
    end
end

# surface buoyancy forcing struct and constructor 

struct BuoyancyForcing
    b★::Float64
    Lx::Float64
    seasonal_amplitude::Float64
    seasonal_period::Float64
    custom_seasonal::Union{Function, Nothing}
end

function BuoyancyForcing(; b★=1.0, Lx=8.0, seasonal_amplitude = 0.0, seasonal_period = 365.0, custom_seasonal = nothing)
    return BuoyancyForcing(b★, Lx, seasonal_amplitude, seasonal_period, custom_seasonal)
end

function make_surface_buoyancy(forcing::BuoyancyForcing, Ny::Int)
    """
    make_surface_buoyancy() : creates surface buoyancy boundary condition function for 2D or 3D sim
    uses BuoyancyForcing struct which contains relevant variables
    """
    
    b★ , Lx = forcing.b★, forcing.Lx

    #use a ternary operator to choose between default seasonal buoyancy forcing or custom buoyancy forcing. 
         #the condition is if custom_seasonal == nothing
    seasonal_forcing(t) = 
        forcing.custom_seasonal === nothing ? 
            (forcing.seasonal_amplitude * (cos(2π * t / forcing.seasonal_period) + 1)/2) : # if there is no custom input this is the default
            (1 + forcing.seasonal_amplitude * forcing.custom_seasonal(t)) # return the custom if there is one

    #old sine forcing
    # if Ny == 1
    #     #2D form of buoyancy forcing : dependent on x, t, p
    #     @inline bˢ_flat(x, t, p) = p.b★ * sin(π * x / p.Lx) * seasonal_forcing(t)
    #     return bˢ_flat
    # else
    #     #3D form of buoyancy forcing : dependent on x, y, t, p
    #     @inline bˢ(x, y, t, p) = p.b★ * sin(π * x / p.Lx) * seasonal_forcing(t)
    #     return bˢ
    # end

    # new tanh forcing

    # base structure of tanh forcing
    bss(x) = (tanh(3 * (x + Lx/3)) - 1 ) / 2

    if Ny == 1
        # 2D tanh with seasonal forcing
        @inline bˢ_flat(x, t, p) = p.b★ * (1 + bss(x) * (1 + seasonal_forcing(t)))
        return bˢ_flat
    else
        #3D form of buoyancy forcing : dependent on x, y, t, p1
        @inline bˢ(x, y, t, p) = p.b★ * (1 + bss(x) * (1 + seasonal_forcing(t)))
        return bˢ
    end
end

# construct the beta-plane configuration of the coriolis parameters

struct CoriolisConfig
    enabled::Bool
    scheme::Symbol #:fplane or :betaplane
    f₀::Float64
    β::Float64
    latitude_south::Float64
    latitude_north::Float64
    reference_latitude::Float64
end

function CoriolisConfig(;
    enabled = false, 
    scheme = :betaplane,
    latitude_south = -90.0, 
    latitude_north = 0.0,
    reference_latitude = nothing
)

    """
    Constructs the coriolis force for the simulation,
    enabled : whether or not to include coriolis force
    scheme : :fplane - constant f or :BetaPlane - traditional beta plane where f varies linearly with latitude_north
                in my simulation x is the meridional direction so f varies with x in the beta plane approx
    latitude_south : the southern latitude boundary in degrees @ x = -Lx/2
    latitude_north : the northern latitude boundary in degrees @ x = Lx/2
    reference_latitude : this is for scheme=:fplane , the reference_latitude 

    BetaPlane approximation - 
        - f(x) = f₀ + β L_x
        - where fₒ is the coriolis parameter evaluated at x = 0 
        - β = df/dx for x axis as the meridional direction
        - we can use earth's rotation rate and radius
            Ω = 7.292115e-5 s⁻¹ 
            R = 6.371e6 m
    """

    if scheme == :betaplane
        #calculate f at domain center 
        φ_center = (latitude_south + latitude_north) / 2
        f₀ = 2 * Ω * sind(φ_center)

        # now we need to calculate β = df/dx where x is the northward direction
        """
        okay so β = df/dφ * dφ/dx = > using chain rule
        dx = R dφ because of arclength relationship
        therfore, β = 2 Ω cos(φ) / R
        """

        φ_rad = deg2rad(φ_center) # convert to radians
        β = (2 * Ω * cos(φ_rad)) / R
        ref_lat = φ_center 

    else #:fplane
        ref_lat  = reference_latitude !== nothing ? reference_latitude :
                    (latitude_south + latitude_north) / 2
        
        f₀ = 2 * Ω * sind(ref_lat)
        β = 0.0 
    end

    return CoriolisConfig(enabled, scheme, f₀, β, latitude_south, latitude_north, ref_lat)
end

function make_coriolis(coriolis_config::CoriolisConfig, domain::DomainConfig)
    """
    this function creates the coriolis object for the HorizontalConvectionSimulation model
        if we don't want rotation it returns nothing 

        for β-plane it adjusts β based on actual domain size to ensure:
            f(x=-Lx/2) = f(latitude_south)
            f(x=Lx/2) = f(latitude_north)
    """
    if !coriolis_config.enabled # if not enabled return nothing for coriolis
        return nothing
    end

    if coriolis_config.scheme == :fplane
        return FPlane(f = coriolis_config.f₀)

    else # :betaplane
        # calculate f at boundaries
        f_south = 2 * Ω * sind(coriolis_config.latitude_south)
        f_north = 2 * Ω * sind(coriolis_config.latitude_north)

        #calculate β from domain
        # f(x) = f₀ + β*xC

        Lx = domain.Lx
        β = (f_north - f_south) / Lx
        f₀ = (f_south + f_north) / 2

        return BetaPlane(f₀ = f₀, β = β)
    end
end

# Wind forcing constructor

struct WindForcing
    enabled::Bool
    τy::Float64
    spatial_profile::Union{Function, Nothing}
    #can add a temporal profile here later if need be
end

function WindForcing(; enabled=false, τy=0.0, 
                        spatial_profile=nothing)

    """
    this function configures the wind stress forcing for the simulation
        I left this for custom wind forcings 

    example: southern westerly wind:
    wind = WindForcing(enabled=true, τy =-1e-4 #westward stress negative y
                        spatial_profile = (x,y) -> exp(-((x - x_center)/width)^2)
                        )
    """
    return WindForcing(enabled, τy, spatial_profile)
end

#here is a helper function for the southern ocean westerlies

function SouthernWesterlies(; domain::DomainConfig, τ_magnitude=1e-4,
                            center_latitude = -50.0, width_degrees=10.0)

    """
    we want our winds to mimic the westerly winds , so they should point in the negative y-direction
        the winds should be confined to the southern hemisphere
        they can be characterized by a gaussian spatial_profile
        neumann boundary conditions

        domain: will give us the latitude Latitude range
        τ_magnitude: peak wind stress magnitude 
        center_latitude : latitude of the peak wind
        width_degrees: the width in degrees of latitude of the gaussian for winds

    """

    #we need to map our latitude to x-coordinates
    #[-Lx/2, Lx/2 ] corresponds to lat_south, lat_north
    lat_south = -90.0
    lat_north = 0.0
    Lx= domain.Lx

    lat_center = (lat_south + lat_north) / 2
    
    #convert center_latitude to x coordinates
    x_center = (center_latitude - lat_center) * Lx / (lat_north - lat_south)

    #convert width from degrees to x-coordinates
    x_width = width_degrees * Lx / (lat_north - lat_south)

    #gaussian spatial profile
    spatial_profile = (x,y) -> exp(-((x-x_center) / x_width)^2)

    return WindForcing(
        enabled = true,
        τy = -τ_magnitude,
        spatial_profile = spatial_profile
    )
end


function make_wind_stress(wind::WindForcing, domain::DomainConfig)
    """
    this function creates the ZONAL wind stress boundary condition
        returns fluxboundaryconditions for v, or nothing if disabled
    """

    if !wind.enabled
        return nothing
    end

    if domain.Ny == 1
        #2D case -- zonal wind requires a 3d simulation
        @warn "Zonal wind forcing requested but Ny=1 (2D simulation), skipping wind forcing..."
        return nothing
    end

    Lx, Ly = domain.Lx, domain.Ly

    #default profiles if not specified
    spatial_fn = wind.spatial_profile === nothing ?
        (x, y) -> 1.0 : wind.spatial_profile

    #3D case
    @inline τy_flux(x, y, t, p) = p.τy * spatial_fn(x,y)

    v_bcs = FieldBoundaryConditions(
        top = FluxBoundaryCondition(τy_flux, parameters=(; τy=wind.τy))
    )

    return v_bcs
end

# Construct the grid

function make_grid(domain::DomainConfig, seafloor_function, architecture)
    """
    Constructs the conputational grid with immersed boundary
    """
    H, Lx, Ly = domain.H, domain.Lx, domain.Ly
    Nx, Ny, Nz = domain.Nx, domain.Ny, domain.Nz

    if Ny  == 1
        #2D Grid with y dimension flat
        underlying_grid = RectilinearGrid(
            architecture, 
            size = (Nx, Nz), 
            x = (-Lx/2, Lx/2), 
            z = (-H, 0), 
            halo = (4, 4),
            topology = (Bounded, Flat, Bounded)
        )
    else
        #3D Grid with y dimension Periodic
        underlying_grid = RectilinearGrid(
            architecture, 
            size = (Nx, Ny, Nz), 
            x = (-Lx/2, Lx/2), 
            y = (-Ly/2, Ly/2), 
            z = (-H, 0), 
            halo = (4, 4, 4), 
            topology = (Bounded, Periodic, Bounded)
        )
    end

    return ImmersedBoundaryGrid(underlying_grid, GridFittedBottom(seafloor_function))
end

# Constructor for the physics parameters

struct PhysicsParams
    Ra::Float64
    Pr::Float64
    b★::Float64
    ν::Float64
    κ::Float64
    advection_scheme::Union{Nothing, Any}
    cfl::Float64
end

function PhysicsParams(; Ra=1e11, Pr=1.0, b★=1.0, H=1.0, advection=true)
    """
    this function calculates the parameters that govern the physics of the problem using the nondimension numbers defined :
        Ra : Rayleigh number is a measure of the relative importance of gravity over viscosity in the momentum equation
        Pr : Prandtl number is a ratio of momentum diffusivity over thermal diffusivity
        
    """
    ν = sqrt(Pr * b★ * H^3 / Ra) # Laplacian viscosity
    κ = ν / Pr #Laplacian diffusivity
    
    if advection
        advection_scheme = WENO()
        cfl = 0.5
    else
        advection_scheme = nothing
        cfl = Inf
    end

    return PhysicsParams(Ra, Pr, b★, ν, κ, advection_scheme, cfl)
end

# Constructor for the initial conditions

function make_initial_buoyancy(b_init, Ny::Int)
    """
    this function creates the initial buoyancy field with small random perturbations
        the input b_init governs the coldstart 
    """

    if Ny == 1
        return (x,z) -> b_init + 1.0e-6 * (randn() - 0.5)
    else
        return (x,y,z) -> b_init + 1.0e-6 * (randn() - 0.5)
    end
end

# configuring the output writer setup

struct OutputConfig
    enabled::Bool
    base_dir::String
    time_interval_fraction::Float64
end

function OutputConfig(; enabled=true, base_dir="/output", time_interval_fraction=200.0)
    return OutputConfig(enabled, base_dir, time_interval_fraction)
end

function setup_output_writers!(simulation, domain, physics, forcing, 
                                output_config, filename_prefix, numhill, h₀_frac, b_init)
    """
    This function configures the output writers for the simulation
    """
    if !output_config.enabled
        return
    end

    model = simulation.model
    τ_eq = sqrt(physics.Ra)
    time_interval = τ_eq / output_config.time_interval_fraction

    #global attributes to output in data file
    global_attributes = Dict(
        "h₀" => h₀_frac, 
        "Ra" => physics.Ra, 
        "Pr" => physics.Pr,
        "ν" => physics.ν, 
        "κ" => physics.κ, 
        "Lx" => domain.Lx, 
        "Ly" => domain.Ly, 
        "H" => domain.H, 
        "b★" => forcing.b★, 
        "b_init" => b_init,
        "Nx" => domain.Nx, 
        "Ny" => domain.Ny, 
        "Nz" => domain.Nz
    )

    #create output directory
    grid_label = "$(domain.Nx)_$(domain.Nz)"
    project_dir = joinpath(output_config.base_dir, grid_label)

    # section indices for 2D slices
    indices = domain.Ny == 1 ? (:,1,:) : (:, domain.Ny÷2, :)

    #get fields and diagnostics
    u, v, w = model.velocities  # unpack velocity fields
    b = model.tracers.b         # 
    bw = @at (Center, Center, Center) b * w

    # oceanostics diagnostics
    ke = KineticEnergyEquation.KineticEnergy(model)
    ε = KineticEnergyEquation.DissipationRate(model)
    χ = TracerVarianceEquation.TracerVarianceDissipationRate(model, :b)
    pe = PotentialEnergy(model)

    # averaged buoyancy
    b_avg_y = Field(Average(b, dims=(2)))

    # checkpointer
    simulation.output_writers[:checkpointer] = Checkpointer(
        model, 
        schedule = TimeInterval(200), 
        dir = project_dir, 
        prefix = string(filename_prefix, "_checkpoint"),
        cleanup = true
    )

    # buoyancy output
    simulation.output_writers[:buoyancy] = NetCDFWriter(
        model, (; b, chi=χ, ∫ϕz = bw), 
        schedule = TimeInterval(time_interval), 
        filename = joinpath(project_dir, "buoyancy.nc"), 
        with_halos = false, 
        global_attributes = global_attributes,
        overwrite_existing = true
    )

    # velocities output
    simulation.output_writers[:velocities] = NetCDFWriter(
        model, (; u, v, w), 
        schedule = TimeInterval(time_interval),
        filename = joinpath(project_dir, "velocities.nc"), 
        with_halos = false, 
        global_attributes = global_attributes,
        overwrite_existing = true
    )

    #section section_snapshots
    simulation.output_writers[:section_snapshots] = NetCDFWriter(
        model, (; b, ke, pe), 
        schedule = TimeInterval(1), 
        indices = indices, 
        filename = joinpath(project_dir, "section_snapshots.nc"), 
        with_halos = false, 
        global_attributes = global_attributes, 
        overwrite_existing = true
    )

    # Zonal time means
    simulation.output_writers[:zonal_time_means] = NetCDFWriter(
        model, (; b=b_avg_y), 
        schedule = AveragedTimeInterval(1, window=1), 
        filename = joinpath(project_dir, "zonal_time_means.nc"), 
        with_halos = false, 
        global_attributes = global_attributes, 
        overwrite_existing = true
    )

    #oceanostics diagnostics 
    simulation.output_writers[:oceanostics] = NetCDFWriter(
        model, (; ke, ε, χ), 
        schedule = TimeInterval(time_interval), 
        indices = indices, 
        filename = joinpath(project_dir, "oceanostics.nc"), 
        with_halos = false, 
        global_attributes = global_attributes, 
        overwrite_existing = true
    )

end

# filename generator 

function generate_filename(advection::Bool, coriolis::Bool, wind::Bool, seasonal_amplitude::Float64, numhill::Int, h₀_frac::Float64,
                            Ra::Float64, b_init::Float64)
    """
    Generates filename prefix describing the simulation input
    """
    runtype = advection ? "turbulent" : "diffusive"

    rotation = coriolis ? "_rotating" : nothing

    external_wind = wind ? "_wind_forced" : nothing

    if seasonal_amplitude!=0.0
        b_time = "_temporalforcing"
    else
        b_time = nothing
    end


    if b_init < 0.0
        starttype = "_coldstart"
    elseif b_init > 0.0
        starttype = "_warmstart"
    else
        starttype = "_zerostart"
    end

    if numhill == 1
        hill_number = "_onehill_"
    elseif numhill == 2
        hill_number = "_twohill_"
    else
        hill_number = "_flat_"
    end

    return string(runtype, rotation, external_wind, b_time, hill_number, h₀_frac, "_Ra", Ra, starttype)
end

######## MAIN SIMULATION Function
function HorizontalConvectionSimulation(;
   #domain parameters
    Nx = 256, 
    Ny = 1,
    Nz = 32, 
    H = 1.0, 
    α = 8.0, 

    #topography parameters
    h₀_frac = 0.6, 
    numhill = 1, 

    #physics parameters
    Ra = 1e11, 
    Pr = 1.0, 
    b★ = 1.0, 
    advection = true, 

    #initial conditions
    b_init = 0.0,

    #buoyancy forcing parameters
    seasonal_amplitude = 0.0,
    seasonal_period = 365.0, 
    custom_seasonal = nothing, 

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
    output_dir = "/Users/hfdrake/code/HorizontalConvection/output/testing/",

    #computational parameters
    architecture = CPU()
)

    # step 1. construct domain using DomainConfig
    domain = DomainConfig(H=H, α=α, Nx=Nx, Ny=Ny, Nz=Nz)

    # step 2. construct seafloor using make_seafloor
    seafloor = make_seafloor(domain, h₀_frac, numhill)

    #step 3. construct the grid using make_grid
    grid = make_grid(domain, seafloor, architecture)

    #step 4. construct physics params using PhysicsParams
    physics = PhysicsParams(Ra=Ra, Pr=Pr, b★=b★, H=H, advection=advection)

    #step 5. construct coriolis 

    coriolis_config = CoriolisConfig(
        enabled = coriolis, 
        scheme = coriolis_scheme, 
        latitude_south = latitude_south, 
        latitude_north = latitude_north, 
        reference_latitude = coriolis_reference_latitude
    )

    coriolis_obj = make_coriolis(coriolis_config, domain)

    #step 6. construct wind forcing

    if use_SO_westerlies
        wind_forcing = SouthernWesterlies(
            domain = domain, 
            τ_magnitude = SO_westerlies_magnitude, 
            center_latitude = SO_westerlies_center_lat,
            width_degrees = SO_westerlies_width
        )

    else 
        #use manual wind configuration
        wind_forcing = WindForcing(
            enabled = wind, 
            τy = wind_stress, 
            spatial_profile = wind_spatial_profile
        )
    end

    v_bcs = make_wind_stress(wind_forcing, domain)

    #step 7. construct the buoyancy forcing with BuoyancyForcing
    forcing = BuoyancyForcing(
        b★=b★, 
        Lx = domain.Lx, 
        seasonal_amplitude = seasonal_amplitude, 
        seasonal_period = seasonal_period, 
        custom_seasonal = custom_seasonal
    )

    surface_buoyancy = make_surface_buoyancy(forcing, domain.Ny)
    b_bcs = FieldBoundaryConditions(
        top = ValueBoundaryCondition(surface_buoyancy, parameters=(; b★, Lx=domain.Lx))
    )

    #combined buoyancy and wind bcs

    boundary_conditions_dic = Dict(:b => b_bcs)
    if v_bcs !== nothing
        boundary_conditions_dic[:v] = v_bcs
    end

    boundary_conditions = NamedTuple(boundary_conditions_dic)

    # step 8. construct the model 

    pressure_solver = ConjugateGradientPoissonSolver(grid)

    model = NonhydrostaticModel(
        grid = grid, 
        advection = physics.advection_scheme, 
        timestepper = :RungeKutta3, 
        tracers = :b, 
        buoyancy = BuoyancyTracer(), 
        coriolis = coriolis_obj,
        closure = ScalarDiffusivity(; physics.ν, physics.κ), 
        hydrostatic_pressure_anomaly = CenterField(grid), 
        pressure_solver = pressure_solver, 
        boundary_conditions = boundary_conditions
    )

    # step 9. set the initial conditions
    B₀ = make_initial_buoyancy(b_init, domain.Ny)
    println("B₀ created: " , typeof(B₀))
    set!(model, b = B₀)

    # step 10. construct the simulation timescale and simulation

    τ_eq = sqrt(Ra)
    min_Δz = minimum_zspacing(grid)
    diffusive_time_scale = min_Δz^2 / physics.κ
    advective_time_scale = sqrt(min_Δz / b★)
    Δt = 0.1 * minimum([diffusive_time_scale, advective_time_scale])

    simulation = Simulation(model, Δt = Δt, stop_time = τ_eq)
    #note ** we need an edit here so that the seasonal cycle has a long tau equilibirum ** 

    #step 11. add timestepper

    wizard = TimeStepWizard(cfl = physics.cfl, diffusive_cfl = 0.2)
    simulation.callbacks[:wizard] = Callback(wizard, IterationInterval(50))

    progress(sim) = @printf("i: % 6d, sim time: % 1.3f, wall_time: % 10s, Δt: % 1.4f, advective CFL: %.2e, diffusive CFL: %.2e\n",
        iteration(sim), time(sim), prettytime(sim.run_wall_time), 
        sim.Δt, AdvectiveCFL(sim.Δt)(sim.model), DiffusiveCFL(sim.Δt)(sim.model)
    )

    simulation.callbacks[:progress] = Callback(progress, IterationInterval(10))

    # step 12. setup the output writers
    filename_prefix = generate_filename(advection, coriolis, wind, seasonal_amplitude, numhill, h₀_frac, Ra, b_init)
    output_config = OutputConfig(
        enabled = output_writer, 
        base_dir = output_dir, 
        time_interval_fraction = 200.0
    )

    setup_output_writers!(
        simulation, domain, physics, forcing, 
        output_config, filename_prefix, numhill, h₀_frac, b_init
    )

    return simulation
end
