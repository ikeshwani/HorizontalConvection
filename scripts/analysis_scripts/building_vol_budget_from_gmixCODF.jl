#the purpose of this script is to analyze the CODF Gmix output and make volume budget plots for one region at a time 

using TopographicHorizontalConvection   # physics: boundary_layer_depth, nearest_xi
using NCDatasets
using CairoMakie
using Statistics
using Printf
using NaNStatistics

#directory containing Ra1e8 hill data
data_dir       = "/work/hdd/bfxn/ikeshwani/HorizontalConvection/output/GPU/GRC/ra1e8_4xstretch_threehill_baseforcing_zerostart/"

#psi b calculation from psi b script, this data has 500 buoyancy bins
psib_file      = joinpath(data_dir, "psi_b_3hill_RA1e8_seg1to23.nc")

#gmix calculation from CODF script, this data has 100 buoyancy bins
#contains gmix calculated from 9 regions: plume, BL, basin 0-3, hill 1-3. but BL is split into 7 interior regions
#also contains M, dMdt, and Residual
#I did find a mistake in this iteration, so im waiting for the rerun. 
gmix_file      = joinpath(data_dir, "Gmix_regions_CODF_3hill_RA1e8_seg1to23.nc")

# getting base data from buoyancy segment 1
ds1 = NCDataset(joinpath(data_dir, "buoyancy_seg1.nc"))
Lx  = Float64(ds1.attrib["Lx"])
Ra  = Float64(ds1.attrib["Ra"])
close(ds1)

#calculating boundary layer depth from TopographicHorizontalConvection:boundary_layer_depth
zBL = boundary_layer_depth(Lx, Ra)

#loading psi_b data from psi_b file itself , b_bins=501
ds_psi = NCDataset(psib_file)
x_psi  = Float64.(ds_psi["x"][:])
b_psi  = Float64.(ds_psi["b"][:])
t_psi  = Float64.(ds_psi["time"][:])
ψ_mean = dropdims(mean(Float64.(ds_psi["psi_b"][:, :, end-10:end]), dims=3), dims=3) # [Nx, n_b]

#load gmix data from gmix CODF calculation file
ds_g  = NCDataset(gmix_file)
t_g   = Float64.(ds_g["time"][:])
b_g   = Float64.(ds_g["b"][:])   # 100 inner bins DOES NOT MATCH length of b_psi

#hill 1 keys: Gmix_hill1, M_hill1, Gmix_bl_hill1, M_bl_hill1, Gsurf_hill1, dMdt_hill1, R_hill1, 
#load hill1 gmix from interior, from boundary layer, and hill one gsurf, take time mean
Gmix_hill1_mean = nanmean(ds_g["Gmix_hill1"][:, end-10:end], dims=2)[:]
Gsurf_hill1_mean = nanmean(ds_g["Gsurf_hill1"][:, end-10:end], dims=2)[:]
GBL_hill1_mean = nanmean(ds_g["Gmix_bl_hill1"][:, end-10:end], dims=2)[:]

#using TopographicHorizontalConvection:nearest_xi find the index closest to the hill one left and right boundaries
index_in_hill1 = nearest_xi(x_psi, -1.35)
index_out_hill1 = nearest_xi(x_psi, -0.65)

#okay i think i finally get the sign convention
#we want to calculate psi which is the transport convergent to the region
# so i want to find the amount coming into the region from the left and the amount coming in from the right 
# want to ensure that n hat points into the region
# with this convention ψ_in_left has convention n hat = positive x hat
# and ψ_in_right has convention n hat = negative x hat
# this makes the total transport = ψ_in_left + ψ_in_right
#then i have to apply a negative sign on both ψ_in_left and ψ_in_right because of the sign convention of psi
#so negative means CCW and positive is CW so we add the negative sign so that a positive transport means to the right and negative means to the left

# using the psi calculated using psi_b file 501 b_bins
# calculate the convergent transport into the hill 1 region
# see notes above on sign convention
ψ_IN_Hill1_left = -ψ_mean[index_in_hill1, :] # the transport INTO hill one from the left side, nhat is in the positive x direction
ψ_IN_Hill1_right = -(-ψ_mean[index_out_hill1, :]) # this is the transport INTO hill one region FROM the RIGHT SIDE, n hat has negative x direction, so we have negative sign
transport_IN = (ψ_IN_Hill1_left .+ ψ_IN_Hill1_right) # delta psi is convergent into the region so the total transport is the SUM OF PSI COMING IN FROM EACH SIDE

transport_from_CODF = nanmean(ds_g["psi_col_hill1"][:, end-10:end], dims=2)[:]
#figure 1: is transport calcuated by hand from psi_b file and calculated in gmix_CODF.jl script consistent?
# we calculated transport by hand from the psi_b file
# the gmix_CODF.jl script calculates transport within the script 

fig1 = Figure()
ax = Axis(fig1[1,1], xlabel= "convergent transport into hill 1", ylabel = "buoyancy", title="ψ into hill 1 from gmix_CODF and psi_b (should be same)") 
lines!(transport_from_CODF, b_g, color=:red, label="CODF transport (100 b bins)")
lines!(transport_IN, b_psi, color=:purple, linestyle=:dot, label="ψ_b transport (501 b bins)")
fig1

#OKAY SO GENERALLY the shape is the same, BUT!!!! the transport calculated in the gmix file has a negative sign!!!
#there is a sign error. and the transport i calculated by hand from the psi_b file is the actually correct one
# i have corrected this in the new gmix_CODF

#now let me calculate the transport BY HAND FROM THE GMIX PSI_B OUTPUT 
#this should correct the incorrect sign error, and theoretically we should get the same result

# i was checking to validate that transport calculated by hand from psi_b from gmix
# but it is not..... this is becaause of the spurious mixing introduced from the size change i believe
# i think it will be fixed in the new iteration of the script
#so the gmix file also calculates psi_b from the same script that the psi_b output file gets it from

ψ_mean_CODF = mean(ds_g["psi_b"][:, :, end-10:end], dims=3)[:, :]
ψ_mean_CODF_hill1_left = - ψ_mean_CODF[index_in_hill1, :]
ψ_mean_CODF_hill1_right = - (-ψ_mean_CODF[index_out_hill1, :])
transport_in_CODF_byHand = ψ_mean_CODF_hill1_left .+ ψ_mean_CODF_hill1_right

#figure 2: is transport calcuated BY HAND from PSI_b file and PSI_b from gmix_CODF consistent?
#comparison figure

fig2 = Figure()
ax = Axis(fig2[1,1], xlabel ="transport", ylabel="buoyancy", title="convergent transport (ψ) into hill 1 calcuated by hand from ψ_b output (gmix_CODF.jl and get_psi_b_sort) (SHOULD BE SAME) ")
lines!(transport_in_CODF_byHand, b_g, color=:"red", label="CODF ψ_b (100 b bins)")
lines!(transport_IN, b_psi, color=:purple, linestyle=:dot, label="ψ_b file (501 b bins)")
axislegend(ax, position=:rb)
fig2

#so now the negative sign error is fixed because i calculated by hand
#BUT the gmix file does some sheesh to make psi_b have the same length as the gmix output
#this causes some spurious effects 
# i have fixed this and hopefully this error will not persist in the new iteration


#FULLLLLL VOLUME BUDGET!

# we don't need to add a negative sign to anything in the figure
# figure three is plotting the transport in from the left, transport in from the right, convergent transport in
# both gmix values added together: gmix in the hill one interior + gmix in hill 1 boundary layer
# gsurf, which is a new calculation! but needed to close the budget

fig3 = Figure()
ax = Axis(fig3[1,1], xlabel ="gmix and transport", ylabel="buoyancy", title="hill one volume budget")
# lines!(Gmix_hill1_mean, b_g, color=:green, linestyle=:solid)
lines!(Gsurf_hill1_mean, b_g, color=:lightgreen, linestyle=:dot, label=L"\mathcal{G}_{BF}")
# lines!(GBL_hill1_mean, b_g, color=:orange, linestyle=:dot)
lines!(Gmix_hill1_mean .+ GBL_hill1_mean, b_g, color=:green, linestyle=:dot, label=L"\mathcal{G}_{mix}")
lines!(ψ_IN_Hill1_left, b_psi, color=:red, label=L"\psi_{in}")
lines!(ψ_IN_Hill1_right, b_psi, color=:lightblue, label=L"\psi_{out}")
lines!(transport_IN, b_psi, color=:purple, linestyle=:dot, label=L"\Delta \psi = \psi_{in} - \psi_{out}")
axislegend(ax, position=:rb)
fig3

#okay this is hard to interpret
# in order to see if my residual is visually small lets plot gmix+g_surf 
#to see if transport ~ -(gmix+g_surf), which would imply a small residual


fig4 = Figure()
ax = Axis(fig4[1,1], xlabel ="gmix and transport", ylabel="buoyancy", title="hill one: gmix + gBF = transport")
lines!(Gmix_hill1_mean .+ GBL_hill1_mean .+ Gsurf_hill1_mean, b_g, color=:green, linestyle=:dot, label=L"\mathcal{G}_{mix} + \mathcal{G}_{BF}")
lines!(ψ_IN_Hill1_left, b_psi, color=:red, label=L"\psi_{in}")
lines!(ψ_IN_Hill1_right, b_psi, color=:lightblue, label=L"\psi_{out}")
lines!(transport_IN, b_psi, color=:purple, linestyle=:dot, label=L"\Delta \psi = \psi_{in} - \psi_{out}")
axislegend(ax, position=:rb)
fig4

#AWESOME
#this looks great, it looks like the residual should be pretty small because dotted green line ~ - dotted purple

# the drift dMdt (though it is really dVdt) is calculated in the gmix file 
dMdt_mean = nanmean(ds_g["dMdt_hill1"][:, end-10:end], dims=2)[:]
M_mean = nanmean(ds_g["M_hill1"][:, end-10:end], dims=2)[:]

# definition of residual : R = dMdt - Gmix_hill_1 - G_mix_bl - G_surf - Transport
# where all the terms are calculated from gmix_CODF.jl
R_hill1_mean = nanmean(ds_g["R_hill1"][:, end-10:end], dims=2)[:]
# this residual is wrong because the transport calculated in the gmix file has two issues
# the first issue is that there is some spurious mixing when psi_b 501 bins is translated to psi_b of length 100
# the second issue is that there is a sign error in the transport psi, such that the residual is double the size that it should be
# both of these edits have been made in gmix_CODF.jl 
# I cannot calculate the residual using the psi_b from the file because the buoyancy axes are different lengths

R_hill1 = dMdt_mean .- Gmix_hill1_mean .- GBL_hill1_mean .- Gsurf_hill1_mean .- transport_IN



#regions = [
#     ("Hill 1",  "Gmix_hill1",  -1.35, -0.65),
#     ("Basin 1", "Gmix_basin1", -0.65, -0.35),
#     ("Hill 2",  "Gmix_hill2",  -0.35,  0.35),
#     ("Basin 2", "Gmix_basin2",  0.35,  0.65),
#     ("Hill 3",  "Gmix_hill3",   0.65,  1.35),
#     ("Basin 3", "Gmix_basin3",  1.35,  x_psi[end]),
# ]
# n_reg = length(regions)   # 6

# # per-region boundary fluxes
# ψ_L_all      = Vector{Vector{Float64}}(undef, n_reg)
# ψ_R_all      = Vector{Vector{Float64}}(undef, n_reg)
# Δψ_all       = Vector{Vector{Float64}}(undef, n_reg)
# g_all        = Vector{Vector{Float64}}(undef, n_reg)
# x_L_used     = Vector{Float64}(undef, n_reg)
# x_R_used     = Vector{Float64}(undef, n_reg)

for (idx, (_, rk, xL, xR)) in enumerate(regions)
    iL = nearest_xi(x_psi, xL);  iR = nearest_xi(x_psi, xR)
    ψ_L_all[idx]  = ψ_inner[iL, :]
    ψ_R_all[idx]  = ψ_inner[iR, :]
    Δψ_all[idx]   = ψ_inner[iR, :] .- ψ_inner[iL, :]
    g_all[idx]    = gmix_means[rk]
    x_L_used[idx] = x_psi[iL]
    x_R_used[idx] = x_psi[iR]
end


ψ_L_all[1]

lines(gmix_means["Gmix_hill1"], b_g)

#check that the volume integrate gmix = 0
ds_sum = NCDataset(gmix_file)
b_sum  = Float64.(ds_sum["b"][:])
t_sum  = Float64.(ds_sum["time"][:])

all_region_keys = [k for k in keys(ds_sum) if startswith(k, "Gmix_")]
println("summing $(length(all_region_keys)) regions: ", all_region_keys)

win = length(t_sum)-10 : length(t_sum)     # last-10-step window (match your other plots)
region_tmeans = Dict(
    k => dropdims(mean(Float64.(ds_sum[k][:, win]), dims=2), dims=2)
    for k in all_region_keys
)
close(ds_sum)

gmix_sum = reduce(+, values(region_tmeans))   # whole-domain G_mix(b), time-mean

maxreg = maximum(maximum(abs, v) for v in values(region_tmeans))
@printf("max|Σ regions| = %.3e   vs   max|any region| = %.3e   (ratio %.1f%%)\n",
        maximum(abs, gmix_sum), maxreg, 100 * maximum(abs, gmix_sum) / maxreg)

fig_sum = Figure(size=(560, 620))
ax = Axis(fig_sum[1, 1], xlabel="⟨G_mix⟩", ylabel="b",
          title="whole-domain check: Σ regions should ≈ 0")
for (_, v) in region_tmeans
    lines!(ax, v, b_sum, color=(:gray, 0.4), linewidth=1)
end
lines!(ax, gmix_sum, b_sum, color=:crimson, linewidth=2.5, label="Σ all regions")
vlines!(ax, 0.0, color=:black, linestyle=:dash, linewidth=0.8)
axislegend(ax, position=:rb)
fig_sum

# =========================================================
# G_surface check: the b≈1 spike in Σ G_mix is the omitted surface
# forcing. Rebuild G_surface (interior mixing stays no-flux) and show
# Σ G_mix + G_surface ≈ 0 pointwise — including at b=1.
#   T_surf = κ (b_surf − b_top)/dz_half · ΔA   (into top cell, like CONV_dV)
#   G_surface(b) = −∂/∂b Σ_{top cells, b_top<b} T_surf   (same −diff as gmix)
# =========================================================
dsg1 = NCDataset(joinpath(data_dir, "buoyancy_seg1.nc"))
Nx, Ny, Nz = dsg1.attrib["Nx"], dsg1.attrib["Ny"], dsg1.attrib["Nz"]
Pr = Float64(dsg1.attrib["Pr"]);  b★ = Float64(dsg1.attrib["b★"]);  H = Float64(dsg1.attrib["H"])
xc = Float64.(dsg1["x_caa"][:])
Δx = Float64.(dsg1["Δx_caa"][:]);  Δy = Float64.(dsg1["Δy_aca"][:]);  Δz = Float64.(dsg1["Δz_aac"][:])
close(dsg1)

κ       = sqrt(Pr * b★ * H^3 / Ra) / Pr
b_surf  = b★ .* tanh.(3 .* (xc .+ Lx/3))          # steady baseforcing surface buoyancy
dz_half = 0.5 * Δz[Nz]                            # top cell center → surface
ΔA      = reshape(Δx, Nx, 1) .* reshape(Δy, 1, Ny)
b_edges = collect(range(-1, 1, length=length(b_sum)+1))   # 101 edges (b_sum = centers)

gsurface_whole(b_top) = begin
    Tsurf = κ .* (reshape(b_surf, Nx, 1) .- b_top) ./ dz_half .* ΔA
    bt = vec(b_top);  tg = vec(Tsurf)
    M  = [sum(@view tg[bt .< b_edges[n]]) for n in 1:length(b_edges)]
    -diff(M) ./ diff(b_edges)
end

# top-row buoyancy over the same last-10 window (from seg23)
dsb = NCDataset(joinpath(data_dir, "buoyancy_seg23.nc"))
tb  = Float64.(dsb["time"][:])
sel = findall(tb .>= t_sum[win[1]] - 1e-9)
isempty(sel) && (sel = max(1, length(tb)-10):length(tb))
b_top_win = Array(dsb["b"][:, :, Nz, sel])
close(dsb)

Gsurf    = dropdims(mean(hcat([gsurface_whole(b_top_win[:, :, k]) for k in 1:length(sel)]...), dims=2), dims=2)
combined = gmix_sum .+ Gsurf

@printf("max|Σ G_mix| = %.3e   max|G_surface| = %.3e   max|sum| = %.3e   (ratio %.1f%%)\n",
        maximum(abs, gmix_sum), maximum(abs, Gsurf), maximum(abs, combined),
        100 * maximum(abs, combined) / maximum(abs, gmix_sum))

fig_gs = Figure(size=(620, 640))
ax_gs = Axis(fig_gs[1, 1], xlabel="⟨transport⟩", ylabel="b",
             title="whole-domain: G_mix (interior) + G_surface should cancel")
lines!(ax_gs, gmix_sum, b_sum, color=:crimson,   linewidth=2.5, label="Σ G_mix (interior)")
lines!(ax_gs, Gsurf,    b_sum, color=:royalblue, linewidth=2.5, label="G_surface")
lines!(ax_gs, combined, b_sum, color=:black,     linewidth=2.5, label="sum (≈ 0?)")
vlines!(ax_gs, 0.0, color=:gray, linestyle=:dash, linewidth=0.8)
axislegend(ax_gs, position=:rb)
fig_gs