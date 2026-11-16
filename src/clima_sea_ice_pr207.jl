# clima_sea_ice_pr207.jl -- ClimaSeaIce PR #207 ("Fix swept-area transport in incremental remapping to match C-grid
# flux", https://github.com/CliMA/ClimaSeaIce.jl/pull/207, open, not merged), loaded over ClimaSeaIce 0.7.2.
#
#     include(joinpath(@__DIR__, "..", "src", "clima_sea_ice_pr207.jl"))   # after `using NumericalEarth`, before the first step
#
# WHAT IT CHANGES. Incremental remapping moves ice across each C-grid face through a swept region. In 0.7.2 the
# region's normal velocity is interpolated from the cell corners, like the tangential one, so the swept area only
# approximates the face flux Δt·uᶠ·Δy the ice dynamics produced, with an error that grows with tangential shear
# (a checkerboard of face velocities averages to zero at the corners and moves no ice at all). The PR takes the
# normal velocity from the face itself, rescaled by Δy/(Δy − Δt·δv/2) on an x-face (Δx/(Δx − Δt·δu/2) on a
# y-face), so the swept area equals the face flux exactly. Only x_face_transport and y_face_transport change; the
# bodies below are 0.7.2's with the PR's diff applied, and nothing else in ClimaSeaIce is touched.
#
# WHY HERE. Sea ice piles to 14-33 m of h·ℵ at the same coastal corners (97E, 103E, 113-117E) in every
# configuration (memory: coastal-ice-pile-is-config-independent), where ice converges against the coast under
# shear: where face and corner velocities differ most.
#
# Guarded to ClimaSeaIce 0.7.2: the methods replace that version's, and must not land on another.

const ClimaSeaIcePackage = NumericalEarth.SeaIces.ClimaSeaIce
pkgversion(ClimaSeaIcePackage) == v"0.7.2" ||
    error("src/clima_sea_ice_pr207.jl patches ClimaSeaIce 0.7.2; this is $(pkgversion(ClimaSeaIcePackage)). Recheck the PR against it.")

@eval ClimaSeaIcePackage begin

@inline function x_face_transport(i, j, k, grid, ::Val{N}, Δt, u, v, ℵ, h, hs, r) where N
    FT = eltype(grid)

    Δy = Δyᶠᶜᶜ(i, j, k, grid)
    xᶜ = Δxᶜᶜᶜ(i, j, k, grid) / 2

    v₁ = ℑxᶠᵃᵃ(i, j,   k, grid, v)
    v₂ = ℑxᶠᵃᵃ(i, j+1, k, grid, v)
    δv = v₂ - v₁

    uᶠ = @inbounds u[i, j, k] * Δy / (Δy - Δt * δv / 2)

    𝔉ℵ = zero(FT)
    𝔉𝓋 = zero(FT)
    𝔉𝓈 = zero(FT)

    rule = quadrature_rule(FT, Val(N))

    for (η, wη) in rule, (s, ws) in rule
        vη = v₁ + η * δv

        x = - s * Δt * uᶠ
        y = - Δy/2 + η * Δy - s * Δt * vη

        𝒥 = Δt * uᶠ * (Δy - s * Δt * δv)

        ℵ̃, h̃, h̃s = reconstruct(i, j, k, grid, x - xᶜ, y, ℵ, h, hs, r)

        w = wη * ws * 𝒥
        𝔉ℵ += w * ℵ̃
        𝔉𝓋 += w * ℵ̃ * h̃
        𝔉𝓈 += w * ℵ̃ * h̃s
    end

    return 𝔉ℵ, 𝔉𝓋, 𝔉𝓈
end

@inline function y_face_transport(i, j, k, grid, ::Val{N}, Δt, u, v, ℵ, h, hs, r) where N
    FT = eltype(grid)

    Δx = Δxᶜᶠᶜ(i, j, k, grid)
    yᶜ = Δyᶜᶜᶜ(i, j, k, grid) / 2

    u₁ = ℑyᵃᶠᵃ(i,   j, k, grid, u)
    u₂ = ℑyᵃᶠᵃ(i+1, j, k, grid, u)
    δu = u₂ - u₁

    vᶠ = @inbounds v[i, j, k] * Δx / (Δx - Δt * δu / 2)

    𝔉ℵ = zero(FT)
    𝔉𝓋 = zero(FT)
    𝔉𝓈 = zero(FT)

    rule = quadrature_rule(FT, Val(N))

    for (ξ, wξ) in rule, (s, ws) in rule
        uξ = u₁ + ξ * δu

        x = - Δx/2 + ξ * Δx - s * Δt * uξ
        y = - s * Δt * vᶠ

        𝒥 = Δt * vᶠ * (Δx - s * Δt * δu)

        ℵ̃, h̃, h̃s = reconstruct(i, j, k, grid, x, y - yᶜ, ℵ, h, hs, r)

        w = wξ * ws * 𝒥
        𝔉ℵ += w * ℵ̃
        𝔉𝓋 += w * ℵ̃ * h̃
        𝔉𝓈 += w * ℵ̃ * h̃s
    end

    return 𝔉ℵ, 𝔉𝓋, 𝔉𝓈
end

end

@info "sea ice: ClimaSeaIce PR #207's incremental-remapping face transport loaded over 0.7.2"
