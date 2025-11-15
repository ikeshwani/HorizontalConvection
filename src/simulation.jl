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
            (1 + forcing.seasonal_amplitude * sin(2π * t / forcing.seasonal_period)) : # if there is no custom input this is the default
            (1 + forcing.seasonal_amplitude * forcing.custom_seasonal(t)) # return the custom if there is one

    if Ny == 1
        #2D form of buoyancy forcing : dependent on x, t, p
        @inline bˢ_flat(x, t, p) = p.b★ * sin(π * x / p.Lx) * seasonal_forcing(t)
        return bˢ_flat
    else
        #3D form of buoyancy forcing : dependent on x, y, t, p
        @inline bˢ(x, y, t, p) = p.b★ * sin(π * x / p.Lx) * seasonal_forcing(t)
        return bˢ
    end
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
        noise(x, z) = 1e-6 * (randn() - 0.5)
        B₀(x, z) = b_init + noise(x, z)
    else
        noise(x, y, z) = 1e-6 * (randn() - 0.5)
        B₀(x, y, z) = b_init + noise(x, y, z)
    end
    return B₀
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

function generate_filename(advection::Bool, numhill::Int, h₀_frac::Float64,
                            Ra::Float64, b_init::Float64)
    """
    Generates filename prefix describing the simulation input
    """
    runtype = advection ? "turbulent" : "diffusive"

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

    return string(runtype, hill_number, h₀_frac, "_Ra", Ra, starttype)
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

    #forcing parameters
    seasonal_amplitude = 0.0,
    seasonal_period = 365.0, 
    custom_seasonal = nothing, 

    #output parameters
    output_writer = true, 
    output_dir = "/Users/hfdrake/code/HorizontalConvection/output/new_pressureSolver/",

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

    #step 5. construct the buoyancy forcing with BuoyancyForcing
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

    # step 6. construct the model 

    pressure_solver = ConjugateGradientPoissonSolver(grid)

    model = NonhydrostaticModel(
        grid = grid, 
        advection = physics.advection_scheme, 
        timestepper = :RungeKutta3, 
        tracers = :b, 
        buoyancy = BuoyancyTracer(), 
        closure = ScalarDiffusivity(; physics.ν, physics.κ), 
        hydrostatic_pressure_anomaly = CenterField(grid), 
        pressure_solver = pressure_solver, 
        boundary_conditions = (; b=b_bcs)
    )

    # step 7. set the initial conditions
    B₀ = make_initial_buoyancy(b_init, domain.Ny)
    set!(model, b = B₀)

    # step 8. construct the simulation timescale and simulation

    τ_eq = sqrt(Ra)
    min_Δz = minimum_zspacing(grid)
    diffusive_time_scale = min_Δz^2 / physics.κ
    advective_time_scale = sqrt(min_Δz / b★)
    Δt = 0.1 * minimum([diffusive_time_scale, advective_time_scale])

    simulation = Simulation(model, Δt = Δt, stop_time = τ_eq)

    #step 9. add timestepper

    wizard = TimeStepWizard(cfl = physics.cfl, diffusive_cfl = 0.2)
    simulation.callbacks[:wizard] = Callback(wizard, IterationInterval(50))

    progress(sim) = @printf("i: % 6d, sim time: % 1.3f, wall_time: % 10s, Δt: % 1.4f, advective CFL: %.2e, diffusive CFL: %.2e\n",
        iteration(sim), time(sim), prettytime(sim.run_wall_time), 
        sim.Δt, AdvectiveCFL(sim.Δt)(sim.model), DiffusiveCFL(sim.Δt)(sim.model)
    )

    simulation.callbacks[:progress] = Callback(progress, IterationInterval(10))

    # step 10. setup the output writers
    filename_prefix = generate_filename(advection, numhill, h₀_frac, Ra, b_init)
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
