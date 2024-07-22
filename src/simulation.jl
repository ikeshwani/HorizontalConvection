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
using Oceananigans.Grids: Center
using Oceananigans.BuoyancyModels: Zᶜᶜᶜ

@inline function PotentialEnergy(model)
    
    b = model.tracers.b
    grid = model.grid
    
    return KernelFunctionOperation{Center, Center, Center}(bz_ccc, grid, b)
end

@inline bz_ccc(i, j, k, grid, b) = - b[i, j, k] * Zᶜᶜᶜ(i, j, k, grid)

function HorizontalConvectionSimulation(; Ra=1e11, h₀_frac=0.6, Nx = 256, Ny = 1, Nz = 32, output_writer=true, advection=true, architecture=CPU())
    
    ## Constant parameters and functions
    H = 1.0            # vertical domain extent
    Lx = 8H            # horizontal domain extent

    if Ny == 1
        Ly = 0.0        # no y dependence y FLAT
    else
        Ly = H/4          
    end

    #Nx, Nz = 256, 32 # meridional, zonal, vertical resolution
    
    Pr = 1.0     # Prandtl number
    
    h₀ = h₀_frac*H
    hill_length = Lx/32

    if Ny != 1
        channel_width = Ly/8
        channel(y) = (1 - (1/3)*exp(-(y^2) / 2channel_width^2))
    elseif Ny == 1
        channel_width = 0.0
        channel(y) = 1.0
    end

    
    hill_1(x) = (2/3)h₀ * exp(-(x-0.0Lx/2)^2 / 2hill_length^2)
    hill_2(x) =      h₀ * exp(-(x-0.5Lx/2)^2 / 2hill_length^2)
    seafloor(x, y) = - H + (hill_1(x) + hill_2(x)) * channel(y)
    
    ## To write a code that loops for two different advection schemes- no advection, and turbulence
    # We write the following for loop - the model will run for both schemes and will print the data 
    # in two different output files
    
    # Define the two different advection schemes, corresponding to turbulent and non-advective physics
    if advection
        advection_scheme = WENO()
        cfl = 0.5
        runtype = "turbulent"
    else
        advection_scheme = nothing
        cfl = Inf
        runtype = "diffusive"
    end
    filename_prefix = string(runtype, "_h", h₀_frac)

    # ### The grid

    underlying_grid_yflat = RectilinearGrid(
            architecture, 
            size = (Nx, Nz)
            x = (-Lx/2, Lx/2), 
            z = (-H, 0), 
            halo = (4, 4), 
            topology = (Bounded, Flat, Bounded))
    

    underlying_grid_yperiodic = RectilinearGrid(
            architecture,
            size = (Nx, Ny, Nz),
            x = (-Lx/2, Lx/2),
            y = (-Ly/2, Ly/2),
            z = (-H, 0),
            halo = (4,4,4),
            topology = (Bounded, Periodic, Bounded))

            
    if Ny == 1
        underlying_grid = underlying_grid_yflat
    elseif Ny != 1
        underlying_grid = underlying_grid_yperiodic
    end

    grid = ImmersedBoundaryGrid(underlying_grid, GridFittedBottom(seafloor))

    
    # ### Boundary conditions
    #
    # At the surface, the imposed buoyancy is
    # ```math
    # b(x, z = 0, t) = b_* \sin (\pi x / L_x) \, ,
    # ```
    # while zero-flux boundary conditions are imposed on all other boundaries. We use free-slip 
    # boundary conditions on ``u`` and ``w`` everywhere.
    
    b★ = 1.0
    @inline bˢ(x, t, p) = p.b★ * sin(π * x / p.Lx)
    
    b_bcs = FieldBoundaryConditions(top = ValueBoundaryCondition(bˢ, parameters=(; b★, Lx)))
    
    # ### Non-dimensional control parameters and Turbulence closure
    #
    # The problem is characterized by three non-dimensional parameters. The first is the domain's
    # aspect ratio, ``H / L_x`` and the other two are the Rayleigh (``Ra``) and Prandtl (``Pr``)
    # numbers:
    #
    # ```math
    # Ra = \frac{b_* H^3}{\nu \kappa} \, , \quad \text{and}\, \quad Pr = \frac{\nu}{\kappa} \, .
    # ```
    #
    # The Prandtl number expresses the ratio of momentum over heat diffusion; the Rayleigh number
    # is a measure of the relative importance of gravity over viscosity in the momentum equation.
    #
    # For a domain with a given extent, the nondimensional values of ``Ra`` and ``Pr`` uniquely
    # determine the viscosity and diffusivity, i.e.,
    # 
    # ```math
    # \nu = \sqrt{\frac{Pr b_* H^3}{Ra}} \quad \text{and} \quad \kappa = \sqrt{\frac{b_* H^3}{Pr Ra}} \, .
    # ```
    #
    # We use isotropic viscosity and diffusivities, `ν` and `κ` whose values are obtain from the
    # prescribed ``Ra`` and ``Pr`` numbers.
    
    ν = sqrt(Pr * b★ * H^3 / Ra)  # Laplacian viscosity
    κ = ν * Pr                     # Laplacian diffusivity
    
    # ## Model instantiation
    #
    # We instantiate the model with the fifth-order WENO advection scheme, a 3rd order
    # Runge-Kutta time-stepping scheme, and a `BuoyancyTracer`.
    
    model = NonhydrostaticModel(; grid,
                                advection = advection_scheme,
                                timestepper = :RungeKutta3,
                                tracers = :b,
                                buoyancy = BuoyancyTracer(),
                                closure = ScalarDiffusivity(; ν, κ),
                                boundary_conditions = (; b=b_bcs))
    
    # ## Simulation set-up
    #
    # We set up a simulation that runs up to ``t = t_f`` with a `JLD2OutputWriter` that saves the flow
    # speed, ``\sqrt{u^2 + w^2}``, the buoyancy, ``b``, and the vorticity, ``\partial_z u - \partial_x w``.

    tf = 100.0
    min_Δz = minimum_zspacing(model.grid)
    diffusive_time_scale = min_Δz^2 / κ
    advective_time_scale = sqrt(min_Δz/b★)
    Δt = 0.1 * minimum([diffusive_time_scale, advective_time_scale])
    simulation = Simulation(model, Δt=Δt, stop_time=tf)
    
    # ### The `TimeStepWizard`
    #
    # The `TimeStepWizard` manages the time-step adaptively, keeping the Courant-Freidrichs-Lewy 
    # (CFL) number close to `0.5` while ensuring the time-step does not increase beyond the 
    # maximum allowable value for numerical stability.
    
    wizard = TimeStepWizard(cfl=cfl, diffusive_cfl=0.2)
    
    simulation.callbacks[:wizard] = Callback(wizard, IterationInterval(50))
    
    # ### A progress messenger
    #
    # We write a function that prints out a helpful progress message while the simulation runs.
    
    progress(sim) = @printf("i: % 6d, sim time: % 1.3f, wall time: % 10s, Δt: % 1.4f, advective CFL: %.2e, diffusive CFL: %.2e\n",
                            iteration(sim), time(sim), prettytime(sim.run_wall_time),
                            sim.Δt, AdvectiveCFL(sim.Δt)(sim.model), DiffusiveCFL(sim.Δt)(sim.model))
    
    simulation.callbacks[:progress] = Callback(progress, IterationInterval(10))
    
    
    # ### Output
    #
    # We use computed `Field`s to diagnose and output the total flow speed, the vorticity, ``\zeta``,
    # and the buoyancy dissipation, ``\chi``. Note that computed `Field`s take "AbstractOperations" on `Field`s as
    # input:
    
    u, v, w = model.velocities # unpack velocity `Field`s
    b = model.tracers.b        # unpack buoyancy `Field`

    # Define online diagnostics
    b_avg_x = Field(Average(b, dims=(2)))
    
    χ = @at (Center, Center, Center) κ * (∂x(b)^2 + ∂z(b)^2)
    
    ke = @at (Center, Center, Center) 1/2 * (u^2 + w^2)
    pe = PotentialEnergy(model)

    # Seed initial buoyancy field with infinitesimal noise,
    # required to break x-symmetry in otherwise x-symmetric configurations!
    noise(x, z) = 1.e-6*(randn()-0.5)
    set!(simulation.model, b=noise);

    # We create a `JLD2OutputWriter` that saves the speed, vorticity, buoyancy dissipation,
    # kineatic energy density, and potential energy density. Because we may want to post-process
    # to post-process prognostic fields in ways that satisfy the boundary conditions,
    # we use the `with_halos = true`.
    #
    # We then add the `JLD2OutputWriter` to the `simulation`.

    if output_writer

    	global_attributes = Dict(
    		"h0" => h₀_frac,
    		"Ra" => Ra,
    	)
		
        simulation.output_writers[:checkpointer] = Checkpointer(
        						model,
        						schedule = TimeInterval(200),
        						dir = "../output",
        						prefix = string(filename_prefix, "_checkpoint"),
        						cleanup = true)

        filename = string("../output/", filename_prefix, "_buoyancy.nc")
        simulation.output_writers[:buoyancy] = NetCDFOutputWriter(model, (; b, chi=χ),
                                                              schedule = TimeInterval(10),
                                                              filename = filename,
                                                              with_halos = true,
                                                              global_attributes = global_attributes,
                                                              overwrite_existing = true)

        filename = string("../output/", filename_prefix, "_velocities.nc")
        simulation.output_writers[:velocities] = NetCDFOutputWriter(model, (; u, w),
                                                              schedule = TimeInterval(100),
                                                              filename = filename,
                                                              with_halos = true,
                                                              global_attributes = global_attributes,
                                                              overwrite_existing = true)

        filename = string("../output/", filename_prefix, "_section_snapshots.nc")
        simulation.output_writers[:section_snapshots] = NetCDFOutputWriter(model, (; b, ke, pe),
                                                              schedule = TimeInterval(1),
                                						      indices = (:,1,:),
                                                              filename = filename,
                                                              with_halos = true,
                                                              global_attributes = global_attributes,
                                                              overwrite_existing = true)

        filename = string("../output/", filename_prefix, "_zonal_time_means.nc")
        simulation.output_writers[:zonal_time_means] = NetCDFOutputWriter(model, (; b=b_avg_x),
                                                              schedule = AveragedTimeInterval(1, window=1),
                                                              filename = filename,
                                                              with_halos = true,
                                                              global_attributes = global_attributes,
                                                              overwrite_existing = true)
        
    end

    return simulation
    
end
