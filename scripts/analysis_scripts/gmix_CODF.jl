using NCDatasets
using CairoMakie
using NaNStatistics

#lets look at one dataset of the Ra1e8 hilly experiment 

data_dir = "/work/hdd/bfxn/ikeshwani/HorizontalConvection/output/GPU/GRC/ra1e8_4xstretch_threehill_baseforcing_zerostart/"
seg = 10
ti = 5
ds = NCDataset(joinpath(data_dir, "buoyancy_seg$(seg).nc"))
ds1 = NCDataset(joinpath(data_dir, "buoyancy_seg1.nc"))

# load in global attribs from seg 1

Nx, Ny, Nz = ds1.attrib["Nx"], ds1.attrib["Ny"], ds1.attrib["Nz"]
Ra, Pr, b★, H = ds1.attrib["Ra"], ds1.attrib["Pr"], ds1.attrib["b★"], ds1.attrib["H"]

x = ds1["x_caa"][:]
y = ds1["y_aca"][:]
z = ds1["z_aac"][:]

Δx_face = ds1["Δx_faa"][:]
Δy_face = ds1["Δy_afa"][:]
Δz_face = ds1["Δz_aaf"][:]

Δx_center = reshape(ds1["Δx_caa"][:], Nx, 1, 1)
Δy_center = reshape(ds1["Δy_aca"][:], 1, Ny, 1)
Δz_center = reshape(ds1["Δz_aac"][:], 1, 1, Nz)

vol = Δx_center .* Δy_center .* Δz_center

ν = sqrt(Pr * b★ * H^3 / Ra)
κ = ν / Pr

#create a wet mask from the second time in segment 1
wet = ds1["b"][:, :, :, 2] .!= 0
close(ds1)

#load in buoyancy from ds analyzing
b = ds["b"][:, :, :, ti]
b[.!wet] .= NaN

# the goal is to recalculate gmix using the convergence of a diffusive flux 
# Gmix = ∂/∂b ∫_(V(b'<b)) -∇ ⋅ (-κ ∇b) dV 

# take finite differences of instantaneous snapshots in buoyancy
# buoyancy lives on the center of the grid, so when we take diff(b), the differences live on the faces

Δb_x = diff(b, dims=1)

# this is our diffusive flux in the x direction, now placed on the FACES and we lost the first face 
flux_x = - κ .* Δb_x ./ reshape(Δx_face[2:Nx], Nx-1, 1, 1)

# we lost the first face, so lets set the flux at face 1 to zero
# also there are 513 faces, so we also need to set the flux at face 512 to zero
flux_x_full = zeros(Nx+1, Ny, Nz)
flux_x_full[2:Nx, :, :] .= flux_x
flux_x_full[isnan.(flux_x_full)] .= 0.0 #flux is zero if dry

# then we take the divergence of this flux in the x dimension : dFdx = ∂Fx/∂x  
dFdx = diff(flux_x_full, dims=1) ./ Δx_center

# multiply by negative 1 to make it the convergence
convX = -1 .* dFdx

# now we repeat for y ... which is periodic! the first face and end face are the same so there are Ny faces not Ny+1
Δb_y = diff(b, dims=2)

# diffusive flux in the y direction:
flux_y = -κ .* Δb_y ./ reshape(Δy_face[2:Ny], 1, Ny-1, 1)

# need to set the flux for cell 1 and have it wrap over to flux of cell end
flux_y_wrap = -κ .* (b[:, 1:1, :] .- b[:, Ny:Ny, :]) ./ Δy_face[1] #this will have size [Nx, 1, Nz] and we need to set it to face 1 = face Ny+1

flux_y_full = zeros(Nx, Ny+1, Nz)
flux_y_full[:, 2:Ny, :] .= flux_y # flux 2:Ny
flux_y_full[:, 1, :] .= flux_y_wrap[:, 1, :] # first face 
flux_y_full[:, Ny+1, :] .= flux_y_wrap[:, 1, :] # last face Ny+1 face
flux_y_full[isnan.(flux_y_full)] .= 0.0 # flux is zero at hill

# divergence of y flux
dFdy = diff(flux_y_full, dims=2) ./ Δy_center

# convergence of y flux 
convY = -1 .* dFdy

# now repeat for z! 
Δb_z = diff(b, dims=3)

flux_z = -κ .* Δb_z ./ reshape(Δz_face[2:Nz], 1, 1, Nz-1)

# z is also bounded so repeat what we did for x (setting flux at faces 1 and Nz+1 = 0)
flux_z_full = zeros(Nx, Ny, Nz+1)
flux_z_full[:, :, 2:Nz] .= flux_z
flux_z_full[isnan.(flux_z_full)] .= 0.0 

# divergence and convergence of flux z 
dFdz = diff(flux_z_full, dims=3) ./ Δz_center
convZ = -1 .* dFdz

# now we have the three vectored components of the convergence of our diffusive flux and we can take the volume integral

CONV = convX .+ convY .+ convZ
CONV_dV = CONV .* vol 
CONV_dV[.!wet] .= 0.0

# we now have to take our cumulative volume integral over V(b'<b), then a partial derivative wrt to b to get Gmix
b_range = (-1.0, 1.0)
n_b_bins = 501
b_edges = collect(range(b_range[1], b_range[2], length=n_b_bins))

keep = vec(wet)
bg = vec(b)[keep] # buoyancy of each wet cell
cg = vec(CONV_dV)[keep] 

# M(b) = ∫_{b'<b} (-∇⋅F) dV
M = zeros(n_b_bins)
for n in 1:n_b_bins
    M[n] = sum(@view cg[bg .< b_edges[n]])
end

# G_mix(b) = ∂M/∂b 
b_centers = 0.5 .* (b_edges[1:end-1] .+ b_edges[2:end])
gmix = diff(M) ./ diff(b_edges)

println("∫(-∇ ⋅ F)dV total = ", sum(cg))

fig = Figure(size=(500, 600))
ax  = Axis(fig[1, 1], xlabel="G_mix(b)  [flux-convergence method]", ylabel="b")
lines!(ax, gmix, b_centers)
vlines!(ax, [0.0], color=:gray, linestyle=:dash)
fig
