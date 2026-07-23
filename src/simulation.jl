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

using NCDatasets
using Oceananigans
using Printf
using Oceananigans.AbstractOperations: KernelFunctionOperation
using Oceananigans.Solvers: ConjugateGradientPoissonSolver
using Oceananigans.Grids: Center
using Oceananigans.BuoyancyFormulations: Zᶜᶜᶜ
using Oceanostics
using CUDA
using Adapt
using TOML
using Dates

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
    x_stretch::Float64
    z_stretch::Float64
end

function DomainConfig(; H= 1.0, α = 8.0, Nx = 256, Ny = 256, Nz = 32, x_stretch=0.24, z_stretch=0.41)
    Lx = α * H
    Ly = Lx/(Nx//Ny) #chapter one configuration is Lx/2!! 
    return DomainConfig(H, Lx, Ly, Nx, Ny, Nz, x_stretch, z_stretch)
end


# construct the seafloor

"""
we are using separate 2D and 3D structs for the seafloor so that it works for GPUs
    first is 2D case Ny==1
"""

struct Seafloor2D
    H::Float64
    Lx::Float64
    h₀_1::Float64
    h₀_2::Float64
    h₀_3::Float64
    hill_length::Float64
end

@inline function (s::Seafloor2D)(x,z)
    hill_1 = s.h₀_1 *
        exp(-(x + s.Lx / 4)^2 / (2 * s.hill_length^2))

    hill_2 = (0.75) * s.h₀_2 *
        exp(-(x)^2 / (2 * s.hill_length^2))

    hill_3 = (0.5) * s.h₀_3 *
        exp(-(x - s.Lx/4)^2 / (2 * s.hill_length^2))

    return -s.H + (hill_1 + hill_2 + hill_3)
end

struct Seafloor3D
    H::Float64
    Lx::Float64
    Ly::Float64
    h₀_1::Float64
    h₀_2::Float64
    h₀_3::Float64
    hill_length::Float64
    channel_width::Float64
end

### now we repeat for the 3D seafloor

@inline function (s::Seafloor3D)(x, y, z)
    hill_1 = s.h₀_1 *
        exp(-(x + s.Lx / 4)^2 / (2 * s.hill_length^2))

    hill_2 = (0.75) * s.h₀_2 *
        exp(-(x)^2 / (2 * s.hill_length^2))

    hill_3 = (0.5) * s.h₀_3 *
        exp(-(x - s.Lx/4)^2 / (2 * s.hill_length^2))

    channel = 1 - (1/3) * exp(-(y^2) / (2 * s.channel_width^2))

    return -s.H + (hill_1 + hill_2 + hill_3) * channel
end

function make_seafloor(domain::DomainConfig, h₀_frac, numhill)
    """
    make_seafloor :
    uses the respective 2D or 3D seafloor structs which contains H, Lx, Ly, h₀_1, h₀_2, hill_length, Channel_length
    h₀_frac : fraction of domain height for hill amplitude
    numhill : Number of hills (0,1, or 2)

    returns seafloor function to input into ImmersedBoundaryGrid

    this is isbits, immutable, GPU-sage
     does not use closures, has no captured variables
    """
    
    H, Lx, Ly = domain.H, domain.Lx, domain.Ly


    if numhill == 1
        h₀_1 = h₀_frac * H
        h₀_2 = 0.0
        h₀_3 = 0.0
    elseif numhill == 2
        h₀_1 = h₀_frac * H 
        h₀_2 = h₀_frac * H
        h₀_3 = 0.0
    elseif numhill == 0 
        h₀_1 = 0.0
        h₀_2 = 0.0
        h₀_3 = 0.0
    elseif numhill == 3
        h₀_1 = h₀_frac * H 
        h₀_2 = h₀_frac * H 
        h₀_3 = h₀_frac * H 

    end

    hill_length = Lx/32
    channel_width = Ly/8

    if domain.Ny == 1
        return Seafloor2D(
            H, 
            Lx, 
            h₀_1, 
            h₀_2,
            h₀_3, 
            hill_length
        )
    else
        return Seafloor3D(
            H, 
            Lx, 
            Ly, 
            h₀_1, 
            h₀_2,
            h₀_3, 
            hill_length, 
            channel_width
        )
    end
end

# lets try getting rid of the surfacebuoyancy2D and 3D structs 
# use plain inline functions like we had before 

struct BuoyancyForcing
    b★::Float64
    Lx::Float64
    winter_amplitude::Float64 
    summer_amplitude::Float64 
    seasonal_period::Float64
end

function BuoyancyForcing(; b★, Lx, winter_amplitude, summer_amplitude, seasonal_period)
    return BuoyancyForcing(b★, Lx, winter_amplitude, summer_amplitude, seasonal_period)
end

@inline function surface_buoyancy_2d(x, t, p)
    # base spatial profile goes from -1 to 1
    b_BASE = (tanh(3 * (x + p.Lx/3)))

    # f is the spatial structure of the function that represents anomaly from base
    f = 0.5 * (1 - tanh(3 * (x + p.Lx/3)))

    # if no seasonal forcing , skip the cosine (so there's no infinity error)
    if p.seasonal_period == 0.0
        return p.b★ * b_BASE
    end

    # S(t) is the seasonal clock: S<0 represents winter, S>0 represents summer
    S = cos(2π * t / p.seasonal_period)

    # we define S_summer and S_winter as positive and negative parts of S
    S_summer = max(S, 0.0)
    S_winter = min(S, 0.0)

    return p.b★ * 
            (b_BASE + 
            f * (p.summer_amplitude * S_summer + 
                 p.winter_amplitude * S_winter)
            )
end


@inline function surface_buoyancy_3d(x, y, t, p)
	# base spatial profile goes from -1 to 1
    b_BASE = (tanh(3 * (x + p.Lx/3)))

    # f is the spatial structure of the function that represents anomaly from base
    f = 0.5 * (1 - tanh(3 * (x + p.Lx/3)))

    # if no seasonal forcing , skip the cosine (so there's no infinity error)
    if p.seasonal_period == 0.0
        return p.b★ * b_BASE
    end

    # S(t) is the seasonal clock: S<0 represents winter, S>0 represents summer
    S = cos(2π * t / p.seasonal_period)

    # we define S_summer and S_winter as positive and negative parts of S
    S_summer = max(S, 0.0)
    S_winter = min(S, 0.0)

    return p.b★ * 
            (b_BASE + 
            f * (p.summer_amplitude * S_summer + 
                 p.winter_amplitude * S_winter)
            )
end

function make_surface_buoyancy(forcing::BuoyancyForcing, Ny::Int)
    """
    make_surface_buoyancy() : creates surface buoyancy boundary condition function for 2D or 3D sim
    uses BuoyancyForcing struct which contains relevant variables

    this is GPU SAFE
    """

    if Ny == 1 
        return surface_buoyancy_2d
    else
        return surface_buoyancy_3d
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

function make_grid(domain::DomainConfig, seafloor_function, architecture, numhill)
    """
    Constructs the conputational grid with OR WITHOUT immersed boundary
    if numhil == 0 (flat bottom), uses regular RectilinearGrid
    otherwise uses ImmersedBoundaryGrid with seafloor

    GPU-compatible !!!!!!!! (hopefully)
    I learned from the testing that I have to pre-compute the seafloor on CPU
    then transfer to GPU
    """
    H, Lx, Ly = domain.H, domain.Lx, domain.Ly
    Nx, Ny, Nz = domain.Nx, domain.Ny, domain.Nz
    x_stretch, z_stretch = domain.x_stretch, domain.z_stretch

    """
    add chebychev grid spacing at for x and z axes
        specifically at the left boundary in x 
        top and bottom in z
    """ 

    # x refined function is pulled out of make grid so lets call it

    function x_refined_grid(i, rx = x_stretch, Lx=Lx, Nx=Nx)
        ξ = (i - 1) / Nx
        β = log(rx) # sets the ratio Δx_max / Δx_min = r

        if rx == 1.0
            stretched = ξ
        else
            stretched = (exp(β*ξ) - 1) / (exp(β) -1)
        end

        return -Lx/2 + Lx * stretched
    end

    #modified z stretch so its only on the top

    function z_refined_grid(k, rz = z_stretch, H=H, Nz=Nz)
        ξ = (k - 1) / Nz
        β = log(rz)

        if rz == 1.0 #need this to avoid divide by 0 error
            stretched = ξ
        else
            stretched = (1 - exp(-β*ξ)) / (1 - exp(-β))
        end

        return -H + H * stretched
    end

    # function left_refined_x_faces(i)
    #     ξ = (i - 1) / Nx 

    #     β = 3.0 * x_stretch
    #     if β > 0.01 
    #         stretched = (exp(β * ξ) - 1) / (exp(β) - 1)
    #     else
    #         stretched = ξ
    #     end

    #     return -Lx/2 + Lx * stretched
    # end

    # function z_chebychev_grid(k)
    #     ξ = (k - 1) / Nz 
    #     cheby = 0.5 * (1 - cos(π * ξ))
    #     uniform = ξ

    #     stretched = (1 - z_stretch) * uniform + z_stretch * cheby

    #     #map to domain [-H, 0]
    #     return -H + H * stretched 
    # end

    if numhill == 0
        if Ny == 1
            return RectilinearGrid(
                architecture, 
                size = (Nx, Nz), 
                x = x_refined_grid, 
                z = z_refined_grid, 
                halo = (4, 4), 
                topology = (Bounded, Flat, Bounded)
            )
        else
            return RectilinearGrid(
                architecture, 
                size = (Nx, Ny, Nz), 
                x = x_refined_grid, 
                y = (-Ly/2, Ly/2), 
                z = z_refined_grid, 
                halo = (4, 4, 4), 
                topology = (Bounded, Periodic, Bounded)
            )
        end
    end

    ### otherwise , create immersed boundary grid with seafloor

    if Ny  == 1
        cpu_grid = RectilinearGrid(
            CPU(), 
            size = (Nx, Nz), 
            x = x_refined_grid, 
            z = z_refined_grid,
            halo = (4, 4), 
            topology = (Bounded, Flat, Bounded)
        )

        #precompute seafloor heights on CPU
        seafloor_heights = zeros(Nx)
        for i in 1:Nx
            x = cpu_grid.xᶜᵃᵃ[i]
            seafloor_heights[i] = seafloor_function(x, 0.0)
        end

        underlying_grid = RectilinearGrid(
            architecture, 
            size = (Nx, Nz), 
            x = x_refined_grid, 
            z = z_refined_grid, 
            halo = (4, 4), 
            topology = (Bounded, Flat, Bounded)
        )

        #use the seafloor heights from the CPU grid but convert 
        if architecture isa GPU
            seafloor_heights = CuArray(seafloor_heights)
        end
       
    else
        #3D Grid with y dimension Periodic
        #Create the CPU grid for computing coords

        cpu_grid = RectilinearGrid(
            CPU(), 
            size = (Nx, Ny, Nz), 
            x = x_refined_grid, 
            y = (-Ly/2, Ly/2), 
            z = z_refined_grid, 
            halo = (4, 4, 4), 
            topology = (Bounded, Periodic, Bounded)
        )

        seafloor_heights = zeros(Nx, Ny)
        for i in 1:Nx , j in 1:Ny
            x = cpu_grid.xᶜᵃᵃ[i]
            y = cpu_grid.yᵃᶜᵃ[j]
            seafloor_heights[i, j] = seafloor_function(x, y, 0.0)
        end

        #create the actual underlying grid
        underlying_grid = RectilinearGrid(
            architecture, 
            size = (Nx, Ny, Nz),
            x = x_refined_grid, 
            y = (-Ly/2, Ly/2), 
            z = z_refined_grid, 
            halo = (4, 4, 4), 
            topology = (Bounded, Periodic, Bounded)
        )

        if architecture isa GPU
            seafloor_heights = CuArray(seafloor_heights)
        end
    end

    return ImmersedBoundaryGrid(underlying_grid, GridFittedBottom(seafloor_heights))
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
        advection_scheme = Centered(order=2)
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

    @inline function noise(x,y, z)
        return 1e-4 * (randn() - 0.5)
    end

    if Ny == 1
        return (x,z) -> b_init + noise(x, 0, z)
    else
        return (x,y,z) -> b_init + noise(x, y, z)
    end
end

# configuring the output writer setup

struct OutputConfig
    enabled::Bool
    base_dir::String
    time_interval_fraction::Float64
end

function OutputConfig(; enabled=true, base_dir="/output", time_interval_fraction=6000.0)
    return OutputConfig(enabled, base_dir, time_interval_fraction)
end

function setup_output_writers!(simulation, domain, physics, forcing, 
                                output_config, filename_prefix, numhill, h₀_frac, b_init, 
                                segment::Int = 1)
    """
    This function configures the output writers for the simulation
    """
    if !output_config.enabled
        return
    end

    model = simulation.model
    τ_eq = sqrt(physics.Ra)
    time_interval = τ_eq / output_config.time_interval_fraction
    time_interval = 0.1

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

    #create output directory: one self-contained folder per run, named by the
    #run prefix.  An empty figures/ subfolder is created once per run (project_dir
    #has no segment in it, so this is per-run, not per-segment).
    project_dir = joinpath(output_config.base_dir, filename_prefix)
    mkpath(joinpath(project_dir, "figures"))

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
    # b_avg_y = Field(Average(b, dims=(2)))

    # checkpointer
    simulation.output_writers[:checkpointer] = Checkpointer(
        model, 
        schedule = TimeInterval(1.0),
        dir = project_dir, 
        prefix = string(filename_prefix, "_checkpoint"),
        cleanup = true
    )

    # buoyancy output
    simulation.output_writers[:buoyancy] = NetCDFWriter(
        model, (; b, chi=χ, ϕz = bw);
        filename = joinpath(project_dir, "buoyancy_seg$(segment).nc"), 
        schedule = TimeInterval(time_interval), 
        with_halos = false, 
        global_attributes = global_attributes,
        overwrite_existing = true
    )

    # velocities output
    simulation.output_writers[:velocities] = NetCDFWriter(
        model, (; u, v, w);
        filename = joinpath(project_dir, "velocities_seg$(segment).nc"),
        schedule = TimeInterval(time_interval),
        with_halos = false, 
        global_attributes = global_attributes,
        overwrite_existing = true
    )

    # #section section_snapshots
    # simulation.output_writers[:section_snapshots] = NetCDFWriter(
    #     model, (; b, ke, pe);
    #     filename = joinpath(project_dir, "section_snapshots_seg$(segment).nc"),
    #     schedule = TimeInterval(10), #make time interval bigger for long run
    #     indices = indices, 
    #     with_halos = false, 
    #     global_attributes = global_attributes, 
    #     overwrite_existing = true
    # )

    # # Zonal time means
    # simulation.output_writers[:zonal_time_means] = NetCDFWriter(
    #     model, (; b=b_avg_y);
    #     filename = joinpath(project_dir, "zonal_time_means_seg$(segment).nc"),
    #     schedule = AveragedTimeInterval(10, window=1), #make time interval larger for long run
    #     with_halos = false, 
    #     global_attributes = global_attributes, 
    #     overwrite_existing = true
    # )

    #oceanostics diagnostics 
    simulation.output_writers[:oceanostics] = NetCDFWriter(
        model, (; ke, pe, ε, χ);
        filename = joinpath(project_dir, "oceanostics_seg$(segment).nc"), 
        schedule = TimeInterval(5), #make time interval larger for long
        indices = indices, 
        with_halos = false, 
        global_attributes = global_attributes, 
        overwrite_existing = true
    )

end

# filename generator 

function generate_filename(advection::Bool, coriolis::Bool, wind::Bool,
                            winter_amplitude::Float64, summer_amplitude::Float64,
                            numhill::Int, h₀_frac::Float64, x_stretch::Float64,
                            Ra::Float64, b_init::Float64)
    """
    Build the run-folder / checkpoint prefix from the headline parameters:
        ra<Ra>_<stretch>xstretch_<topo>_<forcing>_<start>
    e.g. ra1e8_4xstretch_threehill_baseforcing_zerostart

    The full configuration (runtype, h₀_frac, grid size, etc.) is recorded in the
    per-run config.toml — only the headline parameters appear in the folder name.
    """
    # Ra → compact scientific, e.g. 1e8, 1e6, 5e6
    ra_str = replace(@sprintf("%.0e", Ra), "e+0" => "e", "e+" => "e", "e-0" => "e-")

    # stretch ratio → integer when whole (4.0 → "4")
    stretch_str = isinteger(x_stretch) ? string(Int(x_stretch)) : string(x_stretch)

    # topography
    topo = if numhill == 1
        "onehill"
    elseif numhill == 2
        "twohill"
    elseif numhill == 3
        "threehill"
    else
        "flat"
    end

    # forcing: base case = no wind, no rotation, no seasonal cycle
    if !wind && !coriolis && winter_amplitude == 0.0 && summer_amplitude == 0.0
        forcing = "baseforcing"
    else
        mods = String[]
        coriolis && push!(mods, "rotating")
        wind && push!(mods, "windforced")
        (winter_amplitude != 0.0 || summer_amplitude != 0.0) && push!(mods, "seasonal")
        forcing = join(mods, "_")
    end

    # initial condition
    start = b_init < 0.0 ? "coldstart" : (b_init > 0.0 ? "warmstart" : "zerostart")

    return string("ra", ra_str, "_", stretch_str, "xstretch_", topo, "_", forcing, "_", start)
end

function write_run_config(project_dir, run_name; Ra, Pr, b★, ν, κ, advection,
                          Nx, Ny, Nz, H, α, Lx, Ly, x_stretch, z_stretch,
                          numhill, h₀_frac, b_init, winter_amplitude,
                          summer_amplitude, seasonal_period, wind, coriolis,
                          migrated::Bool=false)
    """
    Write a per-run config.toml documenting the full configuration + provenance
    (git SHA, date, Julia version) for reproducibility.  Only writes if the file
    does not already exist, so segment pickups don't clobber the original
    provenance.  `migrated=true` flags a config reconstructed after the fact for a
    pre-existing run (its git_sha/date are the migration's, not the run's).
    """
    config_path = joinpath(project_dir, "config.toml")
    isfile(config_path) && return config_path
    mkpath(project_dir)

    git_sha = try
        readchomp(`git -C $(@__DIR__) rev-parse --short HEAD`)
    catch
        "unknown"
    end

    config = Dict(
        "run" => Dict("name" => run_name),
        "physics" => Dict(
            "Ra" => Ra, "Pr" => Pr, "b_star" => b★,
            "nu" => ν, "kappa" => κ, "advection" => advection,
        ),
        "domain" => Dict(
            "Nx" => Nx, "Ny" => Ny, "Nz" => Nz,
            "H" => H, "alpha" => α, "Lx" => Lx, "Ly" => Ly,
            "x_stretch" => x_stretch, "z_stretch" => z_stretch,
        ),
        "topography" => Dict("numhill" => numhill, "h0_frac" => h₀_frac),
        "forcing" => Dict(
            "b_init" => b_init,
            "winter_amplitude" => winter_amplitude,
            "summer_amplitude" => summer_amplitude,
            "seasonal_period" => seasonal_period,
            "wind" => wind, "coriolis" => coriolis,
        ),
        "provenance" => Dict(
            "git_sha" => git_sha,
            "date" => string(Dates.today()),
            "julia_version" => string(VERSION),
            "migrated" => migrated,
        ),
    )

    open(config_path, "w") do io
        TOML.print(io, config)
    end
    @info "wrote run config → $config_path"
    return config_path
end

######## MAIN SIMULATION Function
function HorizontalConvectionSimulation(;
   #domain parameters
    Nx = 256, 
    Ny = 1,
    Nz = 32, 
    H = 1.0, 
    α = 8.0,
    x_stretch = 0.24,
    z_stretch = 0.41,
    stop_time = 10.0,

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
    winter_amplitude = 0.0,
    summer_amplitude = 0.0, 
    seasonal_period = 365.0,

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
    output_dir = "../output/testing/",
    segment = 1,

    #computational parameters
    architecture = CPU()
)

    # step 1. construct domain using DomainConfig
    domain = DomainConfig(H=H, α=α, Nx=Nx, Ny=Ny, Nz=Nz, x_stretch=x_stretch, z_stretch=z_stretch)

    # step 2. construct seafloor using make_seafloor
    seafloor = make_seafloor(domain, h₀_frac, numhill)

    #step 3. construct the grid using make_grid
    grid = make_grid(domain, seafloor, architecture, numhill)

    if numhill == 0
        @info "grid created : flat bottom (no immersed boundary)
                            left refinement with x_stretch = $x_stretch and top refinement with z_stretch = $z_stretch"

    else
        @info "grid created : immersed boundary with $(numhill) hill(s)
                            left refinement with x_stretch = $x_stretch and top refinment with z_stretch = $z_stretch"
    end

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
        b★ = b★, 
        Lx = domain.Lx, 
        winter_amplitude = winter_amplitude, 
        summer_amplitude = summer_amplitude,
        seasonal_period = seasonal_period
    )

    surface_buoyancy_func = make_surface_buoyancy(forcing, domain.Ny)

    b_bcs = FieldBoundaryConditions(
        top = ValueBoundaryCondition(
            surface_buoyancy_func,
            parameters = (;
                b★ = forcing.b★, 
                Lx = domain.Lx,
                winter_amplitude = forcing.winter_amplitude,
                summer_amplitude = forcing.summer_amplitude,
                seasonal_period = forcing.seasonal_period
            )
        )
    )

    #combined buoyancy and wind bcs

    boundary_conditions_dic = Dict(:b => b_bcs)
    if v_bcs !== nothing
        boundary_conditions_dic[:v] = v_bcs
    end

    boundary_conditions = NamedTuple(boundary_conditions_dic)

    # step 8. construct the model 

    # use ConjugateGradientPoissonSolver only for immersed boundary Grids
    # for regular grid use the default pressure Solvers

    if numhill == 0
        @info "Using default FFT-based pressure solver"
        # no immersed boundary can use the fft based Solver

        if coriolis_obj === nothing
            model = NonhydrostaticModel(
                grid = grid, 
                advection = physics.advection_scheme, 
                timestepper = :RungeKutta3, 
                tracers = :b, 
                buoyancy = BuoyancyTracer(),
                closure = ScalarDiffusivity(; physics.ν, physics.κ), 
                hydrostatic_pressure_anomaly = CenterField(grid), 
                boundary_conditions = boundary_conditions
            )
        else
            model = NonhydrostaticModel(
                grid = grid, 
                advection = physics.advection_scheme, 
                timestepper = :RungeKutta3, 
                tracers = :b, 
                buoyancy = BuoyancyTracer(), 
                coriolis = coriolis_obj,
                closure = ScalarDiffusivity(; physics.ν, physics.κ), 
                hydrostatic_pressure_anomaly = CenterField(grid), 
                boundary_conditions = boundary_conditions
            )
        end
        
    else
        @info "using ConjugateGradient pressure solver for immersed boundary"
        # immersed boundary --> need ConjugateGradientPoissonSolver

        pressure_solver = ConjugateGradientPoissonSolver(grid)

        if coriolis_obj === nothing
            model = NonhydrostaticModel(
                grid = grid, 
                advection = physics.advection_scheme, 
                timestepper = :RungeKutta3, 
                tracers = :b, 
                buoyancy = BuoyancyTracer(),
                closure = ScalarDiffusivity(; physics.ν, physics.κ), 
                hydrostatic_pressure_anomaly = CenterField(grid), 
                pressure_solver = pressure_solver, 
                boundary_conditions = boundary_conditions
            )
        else
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
        end
    end

    # step 9. set the initial conditions
    B₀ = make_initial_buoyancy(b_init, domain.Ny)
    set!(model, b = B₀)

    # step 10. construct the simulation timescale and simulation
    τ_eq = sqrt(Ra)
    min_Δz = minimum_zspacing(grid)
    diffusive_time_scale = min_Δz^2 / physics.κ
    advective_time_scale = sqrt(min_Δz / b★)
    Δt = 0.1 * minimum([diffusive_time_scale, advective_time_scale])

    simulation = Simulation(model, Δt = Δt, stop_time = stop_time + Δt * 10)
    #note ** we need an edit here so that the seasonal cycle has a long tau equilibirum ** 

    # for a purely diffusive simulation, we add zero velocities call back to ensure pure diffusion

    if !advection
        function zero_velocities!(sim)
            fill!(sim.model.velocities.u, 0)
            fill!(sim.model.velocities.v, 0)
            fill(sim.model.velocities.w, 0)
        end
        simulation.callbacks[:zero_velocities] = Callback(zero_velocities!, IterationInterval(1))
    end

    #step 11. add timestepper

    wizard = TimeStepWizard(cfl = physics.cfl, diffusive_cfl = 0.2)
    simulation.callbacks[:wizard] = Callback(wizard, IterationInterval(50))

    # progress(sim) = begin
    #     @printf("i: % 6d, sim time: % 1.3f, wall_time: % 10s, Δt: % 1.4f, advective CFL: %.2e, diffusive CFL: %.2e\n",
    #         iteration(sim), time(sim), prettytime(sim.run_wall_time), 
    #         sim.Δt, AdvectiveCFL(sim.Δt)(sim.model), DiffusiveCFL(sim.Δt)(sim.model)
    # )flush(stdout)
    # end

    progress(sim) = begin
    adv_cfl = AdvectiveCFL(sim.Δt)(sim.model)
    diff_cfl = DiffusiveCFL(sim.Δt)(sim.model)

    adv_cfl  = adv_cfl  === nothing ? 0.0 : adv_cfl
    diff_cfl = diff_cfl === nothing ? 0.0 : diff_cfl

    @printf("i: %6d, sim time: %1.3f, wall_time: %10s, Δt: %1.4f, advective CFL: %.2e, diffusive CFL: %.2e\n",
        iteration(sim),
        time(sim),
        prettytime(sim.run_wall_time),
        sim.Δt,
        adv_cfl,
        diff_cfl
    )

    flush(stdout)
    end

    simulation.callbacks[:progress] = Callback(progress, IterationInterval(10))

    # step 12. setup the output writers
    filename_prefix = generate_filename(advection, coriolis, wind, winter_amplitude, summer_amplitude, numhill, h₀_frac, x_stretch, Ra, b_init)
    output_config = OutputConfig(
        enabled = output_writer,
        base_dir = output_dir,
        time_interval_fraction = 6000.0
    )

    setup_output_writers!(
        simulation, domain, physics, forcing,
        output_config, filename_prefix, numhill, h₀_frac, b_init,
        segment
    )

    # step 13. write a per-run config.toml (full params + provenance) for reproducibility
    if output_writer
        project_dir = joinpath(output_dir, filename_prefix)
        write_run_config(project_dir, filename_prefix;
            Ra=Ra, Pr=Pr, b★=b★, ν=physics.ν, κ=physics.κ, advection=advection,
            Nx=Nx, Ny=Ny, Nz=Nz, H=H, α=α, Lx=domain.Lx, Ly=domain.Ly,
            x_stretch=x_stretch, z_stretch=z_stretch,
            numhill=numhill, h₀_frac=h₀_frac, b_init=b_init,
            winter_amplitude=winter_amplitude, summer_amplitude=summer_amplitude,
            seasonal_period=seasonal_period, wind=wind, coriolis=coriolis)
    end

    return simulation
end
