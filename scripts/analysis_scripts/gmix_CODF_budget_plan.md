# Close the regional volume budget in `gmix_CODF.jl`

> Detailed implementation plan. When approved, a copy of this file will also be
> placed at `scripts/analysis_scripts/gmix_CODF_budget_plan.md` for reference.

## Context

The regional CODF G_mix diagnostic splits the domain into 9 regions
(`gmix_region_masks`, `src/analysis/regions.jl:40`): a plume, **one lumped
boundary layer** spanning all `x ≥ -1.8` above `zBL`, and 7 hill/basin interior
columns that are all capped **below** `zBL`.

The water-mass volume budget we want to close, per region, is

```
∂M/∂t = G_mix + Ψ + G_surface     (all terms volume/time, functions of b)
```

with `M(b,t)` = volume of fluid below buoyancy `b`; `G_mix` the Walin **interior**
diabatic transport `= -∂/∂b ∫_{b'<b} ∇·(κ∇b) dV` (the CODF output). The `∂/∂b`
divides out the `[b]` in `∫ (∇·κ∇b) dV` (`[b]·L³/T`), leaving `L³/T` — a volume
transport, *not* a density. `Ψ = Δψ = ψ_R − ψ_L` is the advective convergence,
where `ψ_b(x,b) = -∫∫_{b'<b} u dy dz` (`get_ψb_sort`,
`src/analysis/streamfunction.jl:55`) integrates the **full column depth**.
`G_surface` is the surface-buoyancy-forcing transformation (see below).

**Two reasons the budget doesn't close as originally written:**

1. **Spatial truncation.** `Ψ` (from `ψ_b`) is full-depth, but interior-column
   `G_mix` stops at `zBL`, so at light (high-`b`) classes the interior has ~no
   `G_mix` while `Ψ ≠ 0`. Fix: subdivide the boundary layer by the same x-cuts as
   the interior columns and add each strip to its column (`G_mix_hill1 +
   G_mix_bl_hill1`). Exact because CODF `G_mix` and volume `M` are **additive over
   the region partition** (CODF is linear: sort → cumsum → one `d/db`; the v2
   product-rule estimator is *not* additive, so build the budget on CODF, not v2).

2. **Missing surface forcing.** `conv_dV_snapshot` uses **no-flux top/bottom**
   (gmix_CODF.jl:107–112), so `G_mix` is *interior mixing only* and the surface
   buoyancy flux is absent. Confirmed empirically: the whole-domain sum of the 9
   region `G_mix` curves is ≈0 in the interior classes but has a large spike at
   `b≈1` — the footprint of the omitted surface forcing (which acts at the surface
   buoyancy values). **Decision: keep `G_mix` as pure interior mixing and add
   `G_surface` as its own explicit budget term** (rather than folding the surface
   flux into `G_mix`), so mixing and forcing stay cleanly separated.

Goal: a per-column closed budget over all buoyancy classes, with explicit
residual `R = ∂M/∂t − G_mix − Ψ − G_surface`, then a dense-class (AABW) zoom.

## Scope / constraints

- **All edits stay in `scripts/analysis_scripts/gmix_CODF.jl`.**
- Do **not** touch v1/v2/v3 in `src/analysis/gmix.jl`; do **not** edit
  `gmix_region_masks` in `src/analysis/regions.jl` (would change the region set
  for the v2/v3 scripts). BL sub-strips are built **locally** in `gmix_CODF.jl`.
- Reuse existing exported helpers `get_ψb_sort`, `boundary_layer_depth`,
  `nearest_xi` — these are streamfunction/region utilities, not gmix estimators.

---

## Grid co-location (required — read first)

The three budget terms land on *different* `b` grids by construction:
`M(b)` and `Ψ(b)` are **cumulative** (`∫_{b'<b}…`) so they naturally evaluate at
bin **edges**; `G_mix(b) = −∂/∂b ∫_{b'<b} D dV` is a finite difference of a
cumulative, so it lands on bin **centers**. The budget closes in the continuum —
the offset is purely numerical — but combining an edges array with a centers
array element-wise injects a **half-bin offset** that shows up as a spurious
residual. So: **every budget term (`G_mix`, `M`, `∂M/∂t`, `Ψ`) must be evaluated
on the same axis, `b_centers`, before anything is added or differenced.**

`G_mix` is already on `b_centers` (`-diff(M_conv)/diff(b_edges)`). `M` and `Ψ`
come from cumsum-and-query sweeps (`volume_below`, `get_ψb_sort`), so simply
**query them at `b_centers`** — exact cumulative values on the right grid, no
interpolation, no offset. (Fallback if reusing an edge-based ψ_b:
`ψ_center[k] = ½(ψ_edge[k] + ψ_edge[k+1])`.) The steps below build this in.

## Step-by-step changes to `gmix_CODF.jl`

### Step 1 — Split the boundary layer into 7 x-strips (local)

Replace the single `"boundary_layer"` region with 7 strips sharing the interior
x-cuts, each restricted to `Z > zBL`. Insert right after line 82–84:

```julia
zBL = boundary_layer_depth(Lx, Ra)
X = reshape(x, :, 1, 1);  Z = reshape(z, 1, 1, :)

# interior column x-bounds (name, xlo, xhi) — mirror gmix_region_masks
col_bounds = [
    ("basin0", -1.8, -1.35), ("hill1", -1.35, -0.65), ("basin1", -0.65, -0.35),
    ("hill2", -0.35, 0.35),  ("basin2", 0.35, 0.65),  ("hill3", 0.65, 1.35),
    ("basin3", 1.35, Inf),
]
# BL strips: same x-bands, above zBL
bl_masks = [("bl_$(nm)", (X .>= xlo) .& (X .< xhi) .& (Z .> zBL))
            for (nm, xlo, xhi) in col_bounds]

# new region list = plume + 7 interior (unchanged) + 7 BL strips (lump dropped)
region_masks = vcat(
    [(nm, m) for (nm, m) in region_masks if nm != "boundary_layer"],
    bl_masks,
)
```

`precompute_regions` (line 83), the region loop (line 183), and the NetCDF save
(line 205) then emit `Gmix_bl_*` automatically — no other edits to the compute
loop. (Replacing the lump avoids double-counting; old lumped BL = sum of the 7.)

### Step 2 — Cumulative volume `M(b,t)` per region (tendency term)

Add next to `gmix_region` (after line 130), mirroring it but accumulating `vol`,
skipping the `d/db`, and **querying at `b_centers`** (co-location requirement):

```julia
# cumulative wet-cell volume below each b_center, over one region's cells
function volume_below(b, idxs)
    bg = vec(b)[idxs]
    vg = vec(vol)[idxs]
    M  = zeros(length(b_centers))
    for n in eachindex(b_centers)
        M[n] = sum(@view vg[bg .< b_centers[n]])
    end
    return M                       # length(b_centers), on b_centers
end
```

Preallocate `M_regions = Dict(name => zeros(Float32, length(b_centers), Nt) ...)`
and fill in the region loop: `M_regions[r.name][:, g] .= volume_below(b, r.idxs)`.
After both passes, `∂M/∂t` by central difference over the (possibly uneven)
`times` vector — still on `b_centers`, so it co-locates with `G_mix`.

### Step 3 — Transport `Ψ` via `ψ_b` (load velocities)

In Pass 2, open `velocities_seg$(s).nc` alongside buoyancy, load
`u = Array(vfile["u"][1:Nx, :, :, t_range])`, and after the segment's G_mix loop
compute `ψ_b` for the segment with the existing helper, then **co-locate onto
`b_centers`** (the helper returns edge values on `b_bins`):

```julia
ψ_seg, ψ_edges = get_ψb_sort(b_seg, u_seg, Δy_vec, Δz_vec, Nx, Ny, Nz, nt;
                             b_range=b_range, n_b_bins=n_b_bins)   # ψ_seg on n_b_bins edges
# average adjacent edges → centers (matches b_centers, length n_b_bins-1)
ψ_seg_c = 0.5 .* (ψ_seg[:, 1:end-1, :] .+ ψ_seg[:, 2:end, :])
ψ_b[:, :, gi] .= ψ_seg_c
```

(`Δy_vec`, `Δz_vec` = the un-reshaped `Δ*_aca`/`Δ*_aac` vectors; `ψ_b` preallocated
as `[Nx, length(b_centers), Nt]`.) Computing `ψ_b` in-file guarantees identical
time steps to `G_mix`/`M`, and the edge→center average removes the half-bin
offset so `Ψ` co-locates with `G_mix` and `∂M/∂t`.

### Step 4 — Surface-forcing transformation `G_surface` per column

Keep `G_mix` as pure interior mixing (no-flux top stays) and add `G_surface`
separately. For the **steady baseforcing** runs the surface buoyancy is
time-independent (`seasonal_period = 0`, simulation.jl:210–211):

```julia
b_surf = b★ .* tanh.(3 .* (x .+ Lx/3))          # [Nx], from global attribs
dz_half = 0.5 * Δz_center[1,1,Nz]                # top cell center → surface
```

Per snapshot, the diffusive-flux transport through each top face (Dirichlet BC,
so the model's surface flux is `κ·(b_surf − b_top)/dz_half`), as a convergence
*into* the top cell — same units/convention as `CONV_dV`:

```julia
b_top  = b[:, :, Nz]                                     # [Nx, Ny] top-row buoyancy
T_surf = κ .* (b_surf .- b_top) ./ dz_half .* ΔA_2d      # [Nx, Ny] transport
```

Then, mirroring `gmix_region` (cumulative binned by the top cell's buoyancy,
one `-∂/∂b`), accumulate over each **column's** top cells (x-range `[xlo,xhi)`,
all y) to get `G_surface_col(b)` on `b_centers`. Store `Gsurf_regions[col]`.
(Surface cells live at `z=0 > zBL`, i.e. in the BL strips, so `G_surface`
attaches to the full column, not the sub-`zBL` interior.)

> Assumes `seasonal_period = 0` (the `baseforcing` runs). If a seasonal run is
> ever analyzed, evaluate `surface_buoyancy_2d(x, t, p)` per time step instead.

### Step 5 — Full-column assembly + residual

For each interior column `c` in `col_bounds` with matching BL strip `bl_c`:

```julia
G_col   = Gmix_regions[c] .+ Gmix_regions["bl_"*c]      # interior mixing, [nb, Nt]
M_col   = M_regions[c]    .+ M_regions["bl_"*c]         # volumes add
Gs_col  = Gsurf_regions[c]                              # surface forcing, [nb, Nt]
iL = nearest_xi(x, xlo);  iR = nearest_xi(x, xhi)       # xhi=x[end] for basin3
Ψ_col   = ψ_b[iR, :, :] .- ψ_b[iL, :, :]               # [nb, Nt]
dMdt    = central_time_diff(M_col, times)              # [nb, Nt]
R_col   = dMdt .- G_col .- Ψ_col .- Gs_col             # residual → ~0
```

Also form time-means over the analysis window (quasi-steady, `∂M/∂t ≈ 0`) for a
clean per-column figure.

### Step 6 — Save + plot

- Extend the NetCDF write (line 200) with: `Gmix_bl_*` (already automatic),
  `M_<name>(b,t)`, `Gsurf_<column>(b,t)`, `psi_b(x,b,t)`, and `R_<column>(b,t)`.
- New figure: per-column budget — `G_col`, `Ψ_col`, `G_surface`, `∂M/∂t`, `R_col`
  vs `b` (`b` on the y-axis, one panel per column), reusing the CODF plotting style.
- Dense-class zoom: same panels restricted to low-`b` bins for the AABW budget.

---

## Sign convention (the part you want to eyeball first)

The budget closes as `∂M/∂t = G_mix + Ψ` **only for one pairing of signs**, and
the CODF code already carries a `-` (`-diff(M)/db`, line 129). Rather than trust
algebra, read the sign off the data:

- **Before approving**, inspect the *existing* CODF output
  (`Gmix_regions_CODF_3hill_RA1e8_seg1to23.nc`): look at
  `⟨G_mix_boundary_layer⟩(b)` — its sign at the light/dense ends tells you the
  transport direction the code assigns. Compare against `Δψ` from the existing
  `psi_b_region_flux.jl` overlay (which already plots `G_mix` vs `Δψ`).
- Once the new residual `R(b)` is computed: `R ≈ 0` ⇒ signs correct;
  `R ≈ 2·Ψ` ⇒ flip `Ψ`; `R ≈ 2·G_mix` ⇒ flip `G_mix`. The residual plot is the
  source of truth and makes the convention self-correcting.

No code guesses the sign — it is exposed and verified.

## Files

- `scripts/analysis_scripts/gmix_CODF.jl` — all edits.
- (post-approval) `scripts/analysis_scripts/gmix_CODF_budget_plan.md` — copy of
  this plan.
- Unchanged, reused: `src/analysis/streamfunction.jl` (`get_ψb_sort`),
  `src/analysis/regions.jl` (`boundary_layer_depth`, `nearest_xi`),
  `velocities_seg*.nc`.

## Verification

Run from `scripts/`: `julia --project=../ analysis_scripts/gmix_CODF.jl`
(hill, Ra1e8). Confirm:
1. Per-region `max|G_mix|` printout (line 231) now lists the `bl_*` strips; their
   sum ≈ the previous lumped `boundary_layer` magnitude.
2. **Whole-domain surface-forcing check:** the whole-domain sum
   `Σ_regions G_mix` (the interior-only `b≈1` spike seen in `scratch.jl`) is
   cancelled by `Σ_columns G_surface` — i.e. `Σ(G_mix + G_surface) ≈ 0` *pointwise*
   across `b`, including at `b≈1`. This validates `G_surface`'s magnitude and sign.
3. Per-column residual `R(b) ≈ 0` within noise across **all** `b` — light classes
   now carry interior+BL `G_mix` + `G_surface` and close.
4. Dense-class zoom shows a closed AABW budget (surface *cooling* term in the
   plume + interior mixing balancing the dense-water transport).
5. (Optional additivity check) sum of 7 `bl_*` + 7 interior + `plume` `G_mix`
   reproduces the whole-domain CODF curve (cross-check vs `G_mix_domain.jl`).
