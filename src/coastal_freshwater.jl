# coastal_freshwater.jl -- the constant coastal freshwater forcing of the nested scripts
# (examples/sowinwarp_nested.jl, sowinwarp_glorys_baseline.jl), from a field file on the nest's grid:
#
#     coastal_freshwater_bsose.nc   tools/coastal_freshwater.jl: B-SOSE's coastal excess, inferred
#                                   (oceFWflx - SFLUX/S_ICE - ERA5 P, minus its offshore value)
#     coastal_freshwater_hj.nc      tools/hammond_jones_freshwater.jl: Hammond & Jones (2016)
#
# Applied as a virtual salt flux in the top cell, -S F / (ρ₀ Δz), the way the coupler applies the rest of
# the freshwater, and constant in time. It goes in through nested_ocean_simulation's `forcing`, which the
# nest ADDS to its boundary relaxation (NestedModels/nested_ocean_model.jl), and leaves the coupled
# surface flux untouched.
#
# OUR LAND. A file whose `on_land` attribute begins with "move" (Hammond & Jones: zero on SOSE's land, so
# whatever falls on ours is real flux on a coastline drawn differently) has each land cell's flux spread
# evenly over the wet cells within `reach` cells (a radius in cells, east-west shortened by cos φ),
# keeping the total. Spread, not put in the nearest cell: that piled 3 rows of moved coast into one
# row at ~58 m/yr, the kind of hotspot the residual field has. Any other file (the B-SOSE residual,
# whose land cells hold copies of their nearest ocean value) has its land cells zeroed, as up to fix9.
#
# Needs Oceananigans, NCDatasets and Printf loaded by the script. Oceananigans names are qualified: not all
# are exported (on_architecture is not), and sowinwarp_glorys_baseline.jl does not load setup_bsose.jl.

const COASTAL_ρ₀ = 1026.0

@inline function coastal_freshwater_dilution(i, j, k, grid, clock, fields, p)
    @inbounds S = fields.S[i, j, k]
    @inbounds F = p.F[i, j]
    return ifelse(k == grid.Nz, -S * F / (p.ρ₀ * Oceananigans.Operators.Δzᶜᶜᶜ(i, j, k, grid)), zero(S))
end

"""
    coastal_freshwater_forcing(file, bottom_height; λ₁, φ₁, scaling, arch, reach = 6)

The `(; S = Forcing(...))` for `nested_ocean_simulation`, from `file` (freshwater_flux, kg m⁻² s⁻¹,
+ into the ocean, on the cell centres λ₁ + (i - 1/2)/scaling, φ₁ + (j - 1/2)/scaling). Logs the Gt/yr
in the file, on our land, moved or zeroed, and applied.
"""
function coastal_freshwater_forcing(file, bottom_height; λ₁, φ₁, scaling, arch, reach = 6)
    isfile(file) || error("$file not found: make it with tools/coastal_freshwater.jl or tools/hammond_jones_freshwater.jl, " *
                          "or set COASTAL_FRESHWATER = false in the switches")
    ds = NCDataset(file)
    lon, lat = Float64.(ds["lon"][:]), Float64.(ds["lat"][:])
    F = Float64.(coalesce.(ds["freshwater_flux"][:, :], 0.0))
    attributes = Dict(ds.attrib)
    close(ds)

    wet = Array(Oceananigans.Fields.interior(bottom_height))[:, :, 1] .< 0
    Nx, Ny = size(wet)
    λc = λ₁ .+ ((1:Nx) .- 0.5) ./ scaling
    φc = φ₁ .+ ((1:Ny) .- 0.5) ./ scaling
    (size(F) == (Nx, Ny) && maximum(abs, lon .- λc) < 1e-6 && maximum(abs, lat .- φc) < 1e-6) ||
        error("$file is not on this grid ($(Nx)x$(Ny), from $(λ₁)E $(φ₁)N at 1/$(scaling) deg)")

    area = [(6.371e6 * deg2rad(1 / scaling))^2 * cosd(φ) for _ in λc, φ in φc]
    gt(G) = sum(G .* area) * 365.25 * 86400 / 1e12
    in_file, on_land = gt(F), gt(ifelse.(wet, 0.0, F))

    move = startswith(lowercase(get(attributes, "on_land", "")), "move")
    G = ifelse.(wet, F, 0.0)
    lost = 0.0
    if move
        for I in CartesianIndices(F)
            (wet[I] || F[I] <= 0) && continue
            i, j = Tuple(I)
            targets = [(ii, jj) for di in -reach:reach, dj in -reach:reach
                       for (ii, jj) in ((i + di, j + dj),)
                       if 1 <= ii <= Nx && 1 <= jj <= Ny && wet[ii, jj] && (di * cosd(φc[j]))^2 + dj^2 <= reach^2]
            if isempty(targets)
                lost += F[I] * area[I]
            else
                ΔF = F[I] * area[I] / sum(area[t...] for t in targets)    # the same rate over every target
                for t in targets
                    G[t...] += ΔF
                end
            end
        end
    end
    lost_gt = lost * 365.25 * 86400 / 1e12

    @info @sprintf("coastal freshwater from %s: %.0f Gt/yr in the file, %.0f on our land %s; %.0f Gt/yr applied, up to %.1f m/yr in a cell (%s)",
                   file, in_file, on_land,
                   move ? @sprintf("(spread over the wet cells within %d cells; %.1f with none dropped)", reach, lost_gt) : "(zeroed)",
                   gt(G), maximum(G) * 365.25 * 86400 / 1000, get(attributes, "method", "no method attribute"))
    get(attributes, "era5_precipitation_subtracted", "true") == "true" ||
        @warn "$file was made without ERA5 precipitation subtracted"

    return (; S = Oceananigans.Forcings.Forcing(coastal_freshwater_dilution; discrete_form = true,
                          parameters = (; F = Oceananigans.Architectures.on_architecture(arch, G), ρ₀ = COASTAL_ρ₀)))
end
