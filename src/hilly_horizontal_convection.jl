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
#
# We start by importing `Oceananigans` and `Printf`.

using Oceananigans
using Printf

## Constant parameters and functions

const H = 1.0            # vertical domain extent
const Lx = 4H            # horizontal domain extent
const Nx, Nz = 2048, 512 # horizontal, vertical resolution

const Pr = 1.0     # Prandtl number
const Ra = 1e12    # Rayleigh number

const h₀ = 0.5*H
const width = 0.05*Lx
hill_1(x) = h₀ * exp(-(x+Lx/8)^2 / 2width^2)
hill_2(x) = 0.75*h₀ * exp(-(x-Lx/4)^2 / 2width^2)
bottom(x,y) = - H + hill_1(x) + hill_2(x)

## To write a code that loops for two different advection schemes- no advection, and turbulence
# We write the following for loop - the model will run for both schemes and will print the data 
# in two different output files

# Define the two different advection schemes, corresponding to turbulent and non-advective physics
advection_schemes = [WENO(), nothing]
cfls = [0.5, Inf]

# Define the respective filenames where data will be stored
filenames = ["turbulent_convection_hills.jld2", "diffusive_convection_hills.jld2"]

for (advection_scheme, filename, cfl) in zip(advection_schemes, filenames, cfls)

# ### The grid

underlying_grid = RectilinearGrid(GPU(),
		       size = (Nx, Nz),
                          x = (-Lx/2, Lx/2),
                          z = (-H, 0),
		       halo = (4,4),
                   topology = (Bounded, Flat, Bounded))

grid = ImmersedBoundaryGrid(underlying_grid, GridFittedBottom(bottom))

# ### Boundary conditions
#
# At the surface, the imposed buoyancy is
# ```math
# b(x, z = 0, t) = b_* \sin (\pi x / L_x) \, ,
# ```
# while zero-flux boundary conditions are imposed on all other boundaries. We use free-slip 
# boundary conditions on ``u`` and ``w`` everywhere.

b★ = 1.0
@inline bˢ(x, y, t, p) = p.b★ * sin(π * x / p.Lx)

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
# prescribed ``Ra`` and ``Pr`` numbers. Here, we use ``Pr = 1`` and ``Ra = 10^8``:

ν = sqrt(Pr * b★ * H^3 / Ra)  # Laplacian viscosity
κ = ν * Pr                     # Laplacian diffusivity
nothing # hide

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

tf = 50.0
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

simulation.callbacks[:progress] = Callback(progress, IterationInterval(50))

# ### Output
#
# We use computed `Field`s to diagnose and output the total flow speed, the vorticity, ``\zeta``,
# and the buoyancy dissipation, ``\chi``. Note that computed `Field`s take "AbstractOperations" on `Field`s as
# input:

u, v, w = model.velocities # unpack velocity `Field`s
b = model.tracers.b        # unpack buoyancy `Field`

χ = @at (Center, Center, Center) κ * (∂x(b)^2 + ∂z(b)^2)


## total flow speed
s = @at (Center, Center, Center) sqrt(u^2 + w^2)

ke = @at (Center, Center, Center) 1/2 * (u^2 + w^2)

pe = @at (Center,Center,Center) -b * model.grid.zᵃᵃᶜ

## y-component of vorticity
ζ = ∂z(u) - ∂x(w)
nothing # hide

# We create a `JLD2OutputWriter` that saves the speed, vorticity, buoyancy dissipation,
# kineatic energy density, and potential energy density. Because we may want to post-process
# to post-process prognostic fields in ways that satisfy the boundary conditions,
# we use the `with_halos = true`.
#
# We then add the `JLD2OutputWriter` to the `simulation`.

simulation.output_writers[:fields] = JLD2OutputWriter(model, (; s, b, ζ, χ, ke, pe),
                                                      schedule = TimeInterval(0.5),
                                                      filename = string("../output/", filename),
                                                      with_halos = true,
                                                      overwrite_existing = true)
nothing # hide

# Ready to press the big red button:

run!(simulation)

end
