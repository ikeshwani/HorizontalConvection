# # Horizontal diffusion example
#
# In "horizontal convection", a non-uniform buoyancy is imposed on top of an initially resting fluid.
# By "horizontal diffusion", we refer to the hypothetical solution of the identically-configured problem 
# but in which there is no buoyancy advection (or turbulence). The difference between the two solutions is
# a useful way of quantifying the degree of turbulence in horizontal convection problems.
#
# This script serves to verify the Nusselt number calculation in the Oceananigans.jl 
# "horizontal_convection" example in the documentation.
#
# ## Install dependencies
#
# First let's make sure we have all required packages installed.

# ```julia
# using Pkg
# pkg"add Oceananigans, CairoMakie"
# ```

# ## Horizontal diffusion
#
# We consider two-dimensional horizontal diffusion problem 
# on the ``(x, z)``-plane (``-L_x/2 \le x \le L_x/2`` and ``-H \le z \le 0``).
# The only forcing on the fluid comes from a prescribed, non-uniform
# buoyancy at the top-surface of the domain.
#
# We start by importing `Oceananigans` and `Printf`.

using Oceananigans
using Printf

H = 1.0          # vertical domain extent
Lx = 2H          # horizontal domain extent
Nx, Nz = 128, 64 # horizontal, vertical resolution

grid = RectilinearGrid(size = (Nx, Nz),
                          x = (-Lx/2, Lx/2),
                          z = (-H, 0),
                   topology = (Bounded, Flat, Bounded))
                   
# ### Boundary conditions
#
# At the surface, the imposed buoyancy is
# ```math
# b(x, z = 0, t) = - b_* \cos (2 \pi x / L_x) \, ,
# ```
# while zero-flux boundary conditions are imposed on all other boundaries. We use free-slip 
# boundary conditions on ``u`` and ``w`` everywhere.

b★ = 1.0

@inline bˢ(x, y, t, p) = - p.b★ * cos(π * x / (p.Lx/2))

b_bcs = FieldBoundaryConditions(top = ValueBoundaryCondition(bˢ, parameters=(; b★, Lx)))

# ### Non-dimensional control parameters and Turbulence closure
#
# The problem is characterized by two non-dimensional parameters. The first is the domain's
# aspect ratio, ``L_x / H`` and the other is the Rayleigh (``Ra``) number:
#
# ```math
# Ra = \frac{b_* L_x^3}{\kappa^2}.
# ```
#
# For a domain with a given extent, the nondimensional value of ``Ra`` determines the diffusivity, i.e.,
# 
# ```math
# \kappa = \sqrt{\frac{b_* L_x^3}{Ra}} \, .
# ```
#
# We use an isotropic diffusivity, `κ` whose value is obtained from the
# prescribed ``Ra`` numbers. Here, we use ``Ra = 10^8``:

Ra = 1e8    # Rayleigh number

κ = sqrt(b★ * Lx^3 / Ra)                  # Laplacian diffusivity
nothing # hide

# ## Model instantiation
#
# We instantiate the model with the fifth-order WENO advection scheme, a 3rd order
# Runge-Kutta time-stepping scheme, and a `BuoyancyTracer`.

model = NonhydrostaticModel(; grid,
                            advection = nothing,
                            timestepper = :RungeKutta3,
                            tracers = :b,
                            buoyancy = BuoyancyTracer(),
                            closure = ScalarDiffusivity(; κ),
                            boundary_conditions = (; b=b_bcs))

# ## Simulation set-up
#
# We set up a simulation that runs up to ``t = 1000`` with a `JLD2OutputWriter` that saves the buoyancy, ``b``.

simulation = Simulation(model, Δt=0.1, stop_time=1000.0)

# ### The `TimeStepWizard`
#
# The `TimeStepWizard` manages the time-step adaptively. Since there is no advection in this problem,
# we set the Courant-Freidrichs-Lewy (CFL) number to Inf since it is irrelevant. A maximum non-dimentional timestep of
# well below 1 ensures that the diffusive CFL condition is satisfied.

wizard = TimeStepWizard(cfl=Inf, max_change=1.25, max_Δt=0.1)

simulation.callbacks[:wizard] = Callback(wizard, IterationInterval(100))

# ### A progress messenger
#
# We write a function that prints out a helpful progress message while the simulation runs.

progress(sim) = @printf("i: % 6d, sim time: % 1.3f, wall time: % 10s, Δt: % 1.4f, diffusive CFL: %.2e\n",
                        iteration(sim), time(sim), prettytime(sim.run_wall_time),
                        sim.Δt, DiffusiveCFL(sim.Δt)(sim.model))

simulation.callbacks[:progress] = Callback(progress, IterationInterval(100))

# ### Output
#
# We use computed `Field`s to diagnose and output the total flow speed, the vorticity, ``\zeta``,
# and the buoyancy, ``b``. Note that computed `Field`s take "AbstractOperations" on `Field`s as
# input:

b = model.tracers.b        # unpack buoyancy `Field`
χ = @at (Center, Center, Center) κ * (∂x(b)^2 + ∂z(b)^2)
nothing # hide

# We create a `JLD2OutputWriter` that saves the speed, and the vorticity. Because we want
# to post-process buoyancy and compute the buoyancy variance dissipation (which is proportional
# to ``|\boldsymbol{\nabla} b|^2``) we use the `with_halos = true`. This way, the halos for
# the fields are saved and thus when we load them as fields they will come with the proper
# boundary conditions.
#
# We then add the `JLD2OutputWriter` to the `simulation`.

saved_output_filename = "horizontal_diffusion.jld2"

simulation.output_writers[:fields] = JLD2OutputWriter(model, (; b, χ),
                                                      schedule = TimeInterval(5),
                                                      filename = saved_output_filename,
                                                      with_halos = true,
                                                      overwrite_existing = true)
nothing # hide

# Ready to press the big red button:

run!(simulation)

# ## Load saved output, process, visualize
#
# We animate the results by loading the saved output, extracting data for the iterations we ended
# up saving at, and ploting the saved fields. From the saved buoyancy field we compute the 
# buoyancy dissipation, ``\chi = \kappa |\boldsymbol{\nabla} b|^2``, and plot that also.
#
# To start we load the saved fields are `FieldTimeSeries` and prepare for animating the flow by
# creating coordinate arrays that each field lives on.

using CairoMakie
using Oceananigans
using Oceananigans.Fields
using Oceananigans.AbstractOperations: volume

saved_output_filename = "horizontal_diffusion.jld2"

## Open the file with our data
b_timeseries = FieldTimeSeries(saved_output_filename, "b")
χ_timeseries = FieldTimeSeries(saved_output_filename, "χ")

times = b_timeseries.times

## Coordinate arrays
xc, yc, zc = nodes(b_timeseries[1])
nothing # hide

χ_timeseries_offline = deepcopy(b_timeseries)

for i in 1:length(times)
  bᵢ = b_timeseries[i]
  χ_timeseries_offline[i] .= @at (Center, Center, Center) κ * (∂x(bᵢ)^2 + ∂z(bᵢ)^2)
end

# Now we're ready to animate using Makie.

@info "Making an animation from saved data..."

n = Observable(1)

title = @lift @sprintf("t=%1.2f", times[$n])

bₙ = @lift interior(b_timeseries[$n], :, 1, :)
χₙ = @lift interior(χ_timeseries[$n], :, 1, :)
χₙ_offline = @lift interior(χ_timeseries_offline[$n], :, 1, :)

blim = 1
χlim = 0.025

axis_kwargs = (xlabel = L"x / H",
               ylabel = L"z / H",
               limits = ((-Lx/2, Lx/2), (-H, 0)),
               aspect = Lx / H,
               titlesize = 20)

fig = Figure(resolution = (600, 1050))

ax_b = Axis(fig[2, 1];
            title = L"buoyancy, $b / b_*$", axis_kwargs...)

ax_χ = Axis(fig[3, 1];
            title = L"buoyancy dissipation, $κ |\mathbf{\nabla}b|^2 \, (L_x / {b_*}^5)^{1/2}$ (online)", axis_kwargs...)

ax_χ_offline = Axis(fig[4, 1];
            title = L"buoyancy dissipation, $κ |\mathbf{\nabla}b|^2 \, (L_x / {b_*}^5)^{1/2}$ (offline)", axis_kwargs...)

fig[1, :] = Label(fig, title, fontsize=24, tellwidth=false)

hm_b = heatmap!(ax_b, xc, zc, bₙ;
                colorrange = (-blim, blim),
                colormap = :thermal)
Colorbar(fig[2, 2], hm_b)

hm_χ = heatmap!(ax_χ, xc, zc, χₙ;
                colorrange = (0, χlim),
                colormap = :dense)
Colorbar(fig[3, 2], hm_χ)

hm_χ_offline = heatmap!(ax_χ_offline, xc, zc, χₙ_offline;
                colorrange = (0, χlim),
                colormap = :dense)
Colorbar(fig[4, 2], hm_χ_offline)

# And, finally, we record a movie.

frames = 1:length(times)

record(fig, "horizontal_diffusion.mp4", frames, framerate=8) do i
    msg = string("Plotting frame ", i, " of ", frames[end])
    print(msg * " \r")
    n[] = i
end
nothing #hide

# ![](horizontal_diffusion.mp4)

# ### Validation of the diffusive solution in Oceananigans.jl

χ_diff_analytical = κ * b★^2 * π * tanh(2π * H / Lx) / (Lx * H)
nothing # hide

# We recover the time from the saved `FieldTimeSeries` and construct an empty arrays to store
# the volume-averaged equilibration ratio (the ratio of the instantaneous volume-integrated buoyancy
# dissipation in a diffusive Oceananigans.jl simulation versus the equilibrium analytical solution.
# The equilibration ratio should asymptote to 1 as the simulation reaches its equilibrium state.

t = b_timeseries.times

Equilibration_ratio, Equilibration_ratio_offline = zeros(length(t)), zeros(length(t))
nothing # hide

# Now we can loop over the fields in the `FieldTimeSeries`, compute kinetic energy and ``Nu``,
# and plot. We make use of `Integral` to compute the volume integral of fields over our domain.

for i = 1:length(t)
    χ_diff_Oceananigans = Field(Integral(χ_timeseries[i] / (Lx * H)))
    compute!(χ_diff_Oceananigans)

    χ_diff_Oceananigans_offline = Field(Integral(χ_timeseries_offline[i] / (Lx * H)))
    compute!(χ_diff_Oceananigans_offline)

    Equilibration_ratio[i] = χ_diff_Oceananigans[1, 1, 1] / χ_diff_analytical # should start at zero and reach 1 when fully equilibrated
    Equilibration_ratio_offline[i] = χ_diff_Oceananigans_offline[1, 1, 1] / χ_diff_analytical # should start at zero and reach 1 when fully equilibrated

  end

fig = Figure(resolution = (850, 600))
 
ax_equil = Axis(
  fig[1, 1],
  xlabel = L"t \, (b_* / L_x)^{1/2}", ylabel = L"$\langle \chi_{Oceananigans} \rangle / \langle \chi_{Analytical} \rangle$ (online)",
  limits=((0,nothing), (0, 10))
)
hlines!(ax_equil, [1], color = :black)
lines!(ax_equil, t, Equilibration_ratio; linewidth = 3)


ax_equil_offline = Axis(
  fig[2, 1],
  xlabel = L"t \, (b_* / L_x)^{1/2}", ylabel = L"$\langle \chi_{Oceananigans} \rangle / \langle \chi_{Analytical} \rangle$ (offline)",
  limits=((0,nothing), (0, 10))
)
hlines!(ax_equil, [1], color = :black)
lines!(ax_equil_offline, t, Equilibration_ratio_offline; linewidth = 3)

save("equilibration_ratio.png", fig, px_per_unit = 2)
current_figure() # hide
fig
