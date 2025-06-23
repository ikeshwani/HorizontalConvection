using Printf
using CairoMakie
using Oceananigans
using Oceananigans.Fields
using Oceananigans: xspacings, yspacings, zspacings

x, y, z = nodes(b)

function ComputeZstar(grid, B)
    x, y, z = nodes(b)
    
    Δx = xspacings(grid, Center())
    Δy = yspacings(grid, Center())
    Δz = zspacings(grid, Center())
    ΔA = Δx * Δy
    ΔV = ΔA * Δz
    
    zstar = zeros(size(B))
    for (i,x0) in enumerate(x)
        for (j,y0) in enumerate(y)
            for (k,z0) in enumerate(z)
                heavyside = B .< reshape(B[i,j,k,:], (1,1,1,size(B,4)))
                zstar[i,j,k,:] = (
                    sum(heavyside .* ΔV, dims=(1,2,3)) ./
                    sum(ones(size(B)[1:2]) .* ΔA)
                ) .+ grid.zᵃᵃᶠ[1]
            end
        end
    end
    
    return zstar
end

f1 = Figure()
ax1 = Axis(f1[1,1],title="Heatmap for zstar", xlabel="x[m]", ylabel="z[m]", limits=((-L/2, L/2), (-H/2, H/2)))
hm = heatmap!(ax1, x, z, zstar[:,1,:,140]; colorrange = (-H/2, H/2))
Colorbar(f1[1, 2], hm);
f1

save("../figures/z_star_heatmap.png", f1)

function PotentialEnergies(grid, B)
    Δx = xspacings(grid, Center())
    Δy = yspacings(grid, Center())
    Δz = zspacings(grid, Center())
    ΔV = Δx * Δy * Δz

    x, y, z = nodes(b)

    z_broadcasted = reshape(z,(1,1,Nz))
    total_potential_energy_offline = sum(-interior(B) .* z_broadcasted .* ΔV, dims=(1,2,3))[1,1,1,:]
    background_potential_energy = sum(-interior(B) .* zstar .* ΔV, dims=(1,2,3))[1,1,1,:]
    available_potential_energy = total_potential_energy .- background_potential_energy

    return total_potential_energy, background_potential_energy, available_potential_energy
end


function KineticPotentialOnline(grid, B)
    KE_timeseries = FieldTimeSeries(filepath, "KE")
    PE_timeseries = FieldTimeSeries(filepath, "PE")

    times = B.times

    kinetic_energy = zeros(size(times))
    potential_energy = zeros(size(times))

    for i = 1:length(times)
        ke_snapshot = Field(Integral(KE_timeseries[i]))
        compute!(ke_snapshot)
        kinetic_energy[i] = ke_snapshot[1,1,1]

        pe_snapshot = Field(Integral(PE_timeseries[i]))
        compute!(pe_snapshot)
        potential_energy[i] = pe_snapshot[1,1,1]
    end

    return potential_energy, kinetic_energy
end


fig_energy = Figure(resolution = (800,500))
ax_KE = Axis(fig_energy[1,1], xlabel = "time (s)", ylabel= "KE (J)", title = "KE versus Time")
lines!(ax_KE, B.times, kinetic_energy)

ax_PE = Axis(fig_energy[2,1], xlabel= "time (s)", ylabel = L"-\int_V(zbdV)", title = "Total PE versus Time")
lines!(ax_PE, B.times, total_potential_energy)
lines!(ax_PE, B.times, total_potential_energy_offline, color="orange", linestyle=:dash)

ax_PEb = Axis(fig_energy[1,2], xlabel="time (s)", ylabel="Background PE", title="Background PE versus Time")
lines!(ax_PEb, B.times, background_potential_energy)

ax_PEa = Axis(fig_energy[2,2], xlabel="time (s)", ylabel="Available PE", title="Available PE versus Time")
lines!(ax_PEa, B.times, available_potential_energy)

fig_energy

save("../figures/energyplots.png", fig_energy)



    