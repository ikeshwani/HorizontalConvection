# gmix_flux_convergence_snapshot.jl
#
# CRUDE, SINGLE-SNAPSHOT practice script for a THIRD estimate of G_mix(b),
# via the convergence of the diffusive buoyancy flux:
#
#     G_mix(b) = d/db  ∫_{V(b' < b)}  -∇·(-κ ∇b)  dV
#              = d/db  ∫_{V(b' < b)}   κ ∇²b       dV
#
# Method (exactly the plan): for ONE instantaneous b snapshot
#   1. finite-difference b in x, y, z  -> diffusive flux F = -κ∇b on cell FACES
#   2. take the finite convergence  -∇·F  into each tracer cell (face fluxes in
#      minus face fluxes out), i.e. the discrete divergence theorem per cell
#   3. mask to wet cells, build the cumulative volume integral over V(b'<b)
#   4. take ONE buoyancy derivative d/db at the very end   <- the only derivative
#      applied to the integrated quantity, so noise stays low
#
# This is deliberately NOT a function and NOT optimized — it is a scratch pad to
# play with. Run from scripts/ with:
#   julia --project=../ analysis_scripts/gmix_flux_convergence_snapshot.jl
#
# Grid notes (Oceananigans NetCDF, stretched in x and z, PERIODIC in y):
#   Δx_caa/Δy_aca/Δz_aac = CELL widths      (centers) -> volumes & face areas
#   Δx_faa/Δy_afa/Δz_aaf = CENTER-TO-CENTER distances (faces)  -> gradients
#   x_faa len Nx+1, z_aaf len Nz+1  (walls at the ends);  y_afa len Ny (periodic)

using NCDatasets
using CairoMakie
using Printf

# ======================= CONFIG — edit and play ==========================
experiment = "hill"      # "hill" (3-hill, immersed boundary) or "control" (flat)
seg        = 20          # which segment file to read
ti         = 5           # which time index INSIDE that file = the one snapshot
b_range    = (-1.0, 1.0) # buoyancy axis for the b-coordinate integral
n_b_bins   = 501         # number of b levels

if experiment == "hill"
    data_dir = "/work/hdd/bfxn/ikeshwani/HorizontalConvection/output/GPU/GRC/ra1e8_4xstretch_threehill_baseforcing_zerostart/"
elseif experiment == "control"
    data_dir = "/work/hdd/bfxn/ikeshwani/HorizontalConvection/output/GPU/GRC/ra1e8_4xstretch_flat_baseforcing_zerostart/"
else
    error("unknown experiment $experiment")
end
# =========================================================================

# ---- load grid + ONE buoyancy snapshot -----------------------------------
ds = NCDataset(joinpath(data_dir, "buoyancy_seg$(seg).nc"))
Nx, Ny, Nz = ds.attrib["Nx"], ds.attrib["Ny"], ds.attrib["Nz"]
Ra, Pr, b★, H = ds.attrib["Ra"], ds.attrib["Pr"], ds.attrib["b★"], ds.attrib["H"]

x = ds["x_caa"][:]
z = ds["z_aac"][:]

Δx_caa = ds["Δx_caa"][:]   # cell widths (len Nx)
Δy_aca = ds["Δy_aca"][:]   # len Ny
Δz_aac = ds["Δz_aac"][:]   # len Nz

Δx_faa = ds["Δx_faa"][:]   # center-to-center, at x-faces (len Nx+1)
Δy_afa = ds["Δy_afa"][:]   # at y-faces (len Ny, periodic)
Δz_aaf = ds["Δz_aaf"][:]   # at z-faces (len Nz+1)

# promote to Float64 so the differencing is clean
b = Array{Float64}(ds["b"][:, :, :, ti])   # ONE snapshot [Nx, Ny, Nz]
close(ds)

# nondimensional diffusivity: ν = sqrt(Pr·b★·H³/Ra),  κ = ν/Pr
ν = sqrt(Pr * b★ * H^3 / Ra)
κ = ν / Pr
@printf("κ = %.3e   (Ra=%.0e, Pr=%g)\n", κ, Ra, Pr)

# ---- wet mask (immersed hills). Solid cells carry b=0; read the mask from
#      seg1 step 2 to avoid the t=0 init zeros. Control is all wet. ----------
if experiment == "hill"
    ds1 = NCDataset(joinpath(data_dir, "buoyancy_seg1.nc"))
    wet = Array(ds1["b"][:, :, :, 2]) .!= 0
    close(ds1)
else
    wet = trues(Nx, Ny, Nz)
end

# Mark solid cells NaN. Then any FACE that straddles a solid cell produces a NaN
# flux, which we set to 0 below = a no-flux (insulating) immersed boundary.
b[.!wet] .= NaN

# reshape helpers so cell widths broadcast over [Nx,Ny,Nz]
Δxi = reshape(Δx_caa, Nx, 1, 1)
Δyj = reshape(Δy_aca, 1, Ny, 1)
Δzk = reshape(Δz_aac, 1, 1, Nz)

# ==========================================================================
# STEP 1+2, direction by direction. For each face we compute the TRANSPORT
# through it = flux · face-area  (units b·L³/T). The per-cell convergence is
# then just (transport in) - (transport out), i.e. the divergence theorem, so
# no extra division by cell volume is needed — we already have ∫_cell(-∇·F)dV.
# ==========================================================================

# ---------------- X  (solid walls at both ends: no-flux) ------------------
# interior x-faces are i = 2..Nx ; diff(b,dims=1)[m] = b[m+1]-b[m] -> face m+1
Fx      = -κ .* diff(b, dims=1) ./ reshape(Δx_faa[2:Nx], Nx-1, 1, 1)  # flux
Tx_int  = Fx .* Δyj .* Δzk                                            # ·(y-z area)
Tx_int[isnan.(Tx_int)] .= 0.0                                        # solid face -> no-flux
Tx = zeros(Nx+1, Ny, Nz)          # face 1 (left wall) & Nx+1 (right wall) stay 0
Tx[2:Nx, :, :] .= Tx_int
convX = -(Tx[2:Nx+1, :, :] .- Tx[1:Nx, :, :])                        # [Nx,Ny,Nz]

# ---------------- Z  (bottom wall + top surface) --------------------------
# interior z-faces k = 2..Nz. NOTE: the TOP surface (face Nz+1) carries the
# imposed buoyancy forcing in the real problem, but here it is left as 0
# (no-flux). If you want the surface diffusive flux, inject it at Tz[:,:,Nz+1].
Fz      = -κ .* diff(b, dims=3) ./ reshape(Δz_aaf[2:Nz], 1, 1, Nz-1)
Tz_int  = Fz .* Δxi .* Δyj
Tz_int[isnan.(Tz_int)] .= 0.0
Tz = zeros(Nx, Ny, Nz+1)          # bottom face 1 & top face Nz+1 stay 0
Tz[:, :, 2:Nz] .= Tz_int
convZ = -(Tz[:, :, 2:Nz+1] .- Tz[:, :, 1:Nz])

# ---------------- Y  (PERIODIC) -------------------------------------------
# interior y-faces j = 2..Ny, plus the wrap face between cell Ny and cell 1.
Fy_int  = -κ .* diff(b, dims=2) ./ reshape(Δy_afa[2:Ny], 1, Ny-1, 1)
Ty_int  = Fy_int .* Δxi .* Δzk
Ty_int[isnan.(Ty_int)] .= 0.0
# wrap face: gradient from cell Ny -> cell 1
Fy_wrap = -κ .* (b[:, 1:1, :] .- b[:, Ny:Ny, :]) ./ Δy_afa[1]
Ty_wrap = Fy_wrap .* Δxi .* Δzk
Ty_wrap[isnan.(Ty_wrap)] .= 0.0
Ty = zeros(Nx, Ny+1, Nz)
Ty[:, 2:Ny, :] .= Ty_int
Ty[:, 1,    :] .= Ty_wrap[:, 1, :]   # face 1  == wrap face
Ty[:, Ny+1, :] .= Ty_wrap[:, 1, :]   # face Ny+1 is the same physical face
convY = -(Ty[:, 2:Ny+1, :] .- Ty[:, 1:Ny, :])

# ---- total per-cell convergence  ∫_cell (-∇·F) dV ------------------------
CV = convX .+ convY .+ convZ         # [Nx,Ny,Nz], units b·L³/T
CV[.!wet] .= 0.0                     # solid cells contribute nothing

# ==========================================================================
# STEP 3+4: cumulative volume integral over V(b'<b), then ONE d/db.
# ==========================================================================
b_edges = collect(range(b_range[1], b_range[2], length=n_b_bins))

# keep only wet cells (drop the NaN'd solid ones) before binning
keep = vec(wet)
bg   = vec(b)[keep]     # buoyancy of each wet cell
cg   = vec(CV)[keep]    # its convergence·volume

# M(b) = ∫_{b'<b} (-∇·F) dV  — cumulative integral, crude bin-by-bin sum
M = zeros(n_b_bins)
for n in 1:n_b_bins
    M[n] = sum(@view cg[bg .< b_edges[n]])
end

# G_mix(b) = dM/db — the single derivative, applied at the very end
b_centers = 0.5 .* (b_edges[1:end-1] .+ b_edges[2:end])
Gmix      = diff(M) ./ diff(b_edges)

@printf("total interior convergence  ∫(-∇·F)dV = %.3e  (should be ~0 for closed no-flux domain)\n", sum(cg))
@printf("G_mix:  min=%.3e  max=%.3e\n", minimum(Gmix), maximum(Gmix))

# ---- quick look --------------------------------------------------------
fig = Figure(size=(500, 600))
ax  = Axis(fig[1, 1], xlabel="G_mix(b)  [flux-convergence method]", ylabel="b",
           title="$(experiment)  seg$(seg) ti$(ti)")
lines!(ax, Gmix, b_centers)
vlines!(ax, [0.0], color=:gray, linestyle=:dash)
outpng = joinpath(data_dir, "figures", "gmix_fluxconv_$(experiment)_seg$(seg)_ti$(ti).png")
mkpath(dirname(outpng))
save(outpng, fig)
println("saved figure → $outpng")