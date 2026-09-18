# ==============================================================================
# analyze_velocity_diagnostics.jl
#
# Analyzes NetCDF output from simulation runs to identify where and when
# velocity pileups occur. Useful for diagnosing numerical instabilities.
#
# Usage:
#   julia analyze_velocity_diagnostics.jl [surface_file] [mid_lon_file]
#
# Default file paths:
#   - model_surface_fields.nc
#   - model_mid_lon.nc
# ==============================================================================

using NCDatasets
using Statistics
using Printf

function print_ncdf_info(ds, label)
    """Print NetCDF structure information."""
    println("    Variables in $label file:")
    for (name, var) in ds
        size_str = string(size(var))
        println("      - $name: $size_str")
    end
end

function load_datasets(surface_file="model_surface_fields.nc", mid_lon_file="model_mid_lon.nc")
    """Load NetCDF files and return datasets."""
    println("Loading NetCDF files...")

    ds_surface = nothing
    ds_mid = nothing

    if isfile(surface_file)
        ds_surface = NCDataset(surface_file)
        println("  ✓ Loaded $surface_file")
        print_ncdf_info(ds_surface, "surface")
    else
        println("  ✗ $surface_file not found")
    end

    if isfile(mid_lon_file)
        ds_mid = NCDataset(mid_lon_file)
        println("  ✓ Loaded $mid_lon_file")
        print_ncdf_info(ds_mid, "mid-lon")
    else
        println("  ✗ $mid_lon_file not found")
    end

    if ds_surface === nothing && ds_mid === nothing
        error("No NetCDF files found!")
    end

    return ds_surface, ds_mid
end

function analyze_surface_velocities(ds_surface)
    """Analyze surface velocity statistics and identify spike location."""
    if ds_surface === nothing
        return nothing
    end

    println("\n" * "="^80)
    println(" SURFACE VELOCITY ANALYSIS")
    println("="^80)

    # Handle 4D variables (staggered grids from Oceananigans)
    u_var = ds_surface["u"]
    v_var = ds_surface["v"]
    times = ds_surface["time"][:]

    # Load data: u is (361, 150, 1, 48), v is (360, 151, 1, 48)
    u_4d = u_var[:, :, 1, :]
    v_4d = v_var[:, :, 1, :]

    # Interpolate to cell centers by averaging across staggered dimensions:
    # u on x-faces: average to get (360, 150, 48)
    # v on y-faces: average to get (360, 150, 48)
    u_cells = 0.5 .* (u_4d[1:end-1, :, :] .+ u_4d[2:end, :, :])
    v_cells = 0.5 .* (v_4d[:, 1:end-1, :] .+ v_4d[:, 2:end, :])

    # Compute velocity magnitude at cell centers
    speed = @. sqrt(u_cells^2 + v_cells^2)

    println("\nGrid dimensions: $(size(u_cells)[1]) x $(size(u_cells)[2]) (horizontal, at cell centers)")
    println("Time steps: $(size(u_cells)[3])")

    # Statistics by time
    println("\n" * "-"^80)
    println("Velocity Statistics by Time Step (Sampled)")
    println("-"^80)
    @printf("%-15s %-15s %-15s %-15s %-15s\n", "Time (days)", "Max Speed", "Mean Speed", "P99", "P95")
    println("-"^80)

    # Sample every ~5% of time steps for overview
    sample_indices = round.(Int, range(1, size(speed, 3), length=min(20, size(speed, 3))))

    max_speeds_all = Float64[]

    for i in axes(speed, 3)
        s = speed[:, :, i]
        max_s = maximum(s)
        push!(max_speeds_all, max_s)
    end

    for i in sample_indices
        time_days = times[i]
        s = speed[:, :, i]
        max_speed = maximum(s)
        mean_speed = mean(s)
        p99_speed = quantile(vec(s), 0.99)
        p95_speed = quantile(vec(s), 0.95)

        @printf("%-15.4f %-15.6f %-15.6f %-15.6f %-15.6f\n",
                time_days, max_speed, mean_speed, p99_speed, p95_speed)
    end

    # Find when velocity spike occurs
    time_of_spike = argmax(max_speeds_all)
    spike_value = max_speeds_all[time_of_spike]

    # Detect if there's a significant jump in velocities
    if time_of_spike > 1
        prev_max = maximum(max_speeds_all[1:time_of_spike-1])
        spike_jump = spike_value / prev_max
    else
        prev_max = NaN
        spike_jump = NaN
    end

    println("\n" * "-"^80)
    println("VELOCITY SPIKE DETECTION")
    println("-"^80)
    @printf("Time of spike        : %.4f days (index %d)\n", times[time_of_spike], time_of_spike)
    @printf("Max surface speed    : %.6f m/s\n", spike_value)
    if !isnan(spike_jump)
        @printf("Jump factor          : %.2f x (from %.6f m/s)\n", spike_jump, prev_max)
    end
    println("-"^80)

    # Spatial analysis at time of spike
    println("\nSpatial distribution at spike time:")
    s_spike = speed[:, :, time_of_spike]
    n_cells_gt_1 = sum(s_spike .> 1.0)
    n_cells_gt_5 = sum(s_spike .> 5.0)
    n_cells_gt_10 = sum(s_spike .> 10.0)
    total_cells = length(s_spike)

    @printf("  - Cells with speed > 1 m/s  : %8d / %d (%.3f%%)\n",
            n_cells_gt_1, total_cells, 100*n_cells_gt_1/total_cells)
    @printf("  - Cells with speed > 5 m/s  : %8d / %d (%.3f%%)\n",
            n_cells_gt_5, total_cells, 100*n_cells_gt_5/total_cells)
    @printf("  - Cells with speed > 10 m/s : %8d / %d (%.3f%%)\n",
            n_cells_gt_10, total_cells, 100*n_cells_gt_10/total_cells)

    # Classify the problem
    println("\nProblem classification:")
    if n_cells_gt_10 == 0
        println("  ⊙ No high-speed pileup detected")
    elseif n_cells_gt_10 < total_cells * 0.01
        println("  ⚠ LOCALIZED pileup: high speeds concentrated in <1% of domain")
        # Find bounding box of high speeds
        indices = findall(s_spike .> 10.0)
        if !isempty(indices)
            x_coords = [idx[1] for idx in indices]
            y_coords = [idx[2] for idx in indices]
            println("    High-speed region indices:")
            @printf("      x (lon): %d to %d (range: %d cells)\n", minimum(x_coords), maximum(x_coords), maximum(x_coords) - minimum(x_coords) + 1)
            @printf("      y (lat): %d to %d (range: %d cells)\n", minimum(y_coords), maximum(y_coords), maximum(y_coords) - minimum(y_coords) + 1)

            # Find peak location
            max_idx = argmax(s_spike)
            @printf("    Peak speed location: x=%d, y=%d (speed = %.4f m/s)\n",
                    max_idx[1], max_idx[2], s_spike[max_idx])
        end
    elseif n_cells_gt_10 < total_cells * 0.1
        println("  ⚠ REGIONAL pileup: high speeds in 1-10% of domain")
    else
        println("  ⚠ WIDESPREAD pileup: high speeds in >10% of domain")
    end

    return max_speeds_all, times
end

function analyze_mid_lon_velocities(ds_mid)
    """Analyze mid-longitude vertical slice velocities."""
    if ds_mid === nothing
        return
    end

    println("\n" * "="^80)
    println(" MID-LONGITUDE SLICE VELOCITY ANALYSIS")
    println("="^80)

    # Handle 4D variables (staggered grids from Oceananigans)
    u_var = ds_mid["u"]
    v_var = ds_mid["v"]

    # Load data: u is (1, 150, 135, 48), v is (1, 151, 135, 48)
    u_4d = u_var[:, :, :, :]
    v_4d = v_var[:, :, :, :]
    times = ds_mid["time"][:]

    # Interpolate v to cell centers (average in y direction)
    # u: (1, 150, 135, 48) at center-y
    # v: (1, 151, 135, 48) at face-y -> average to (1, 150, 135, 48)
    u_cells = u_4d
    v_cells = 0.5 .* (v_4d[:, 1:end-1, :, :] .+ v_4d[:, 2:end, :, :])

    # Compute horizontal speed
    speed_h = @. sqrt(u_cells^2 + v_cells^2)

    println("\nGrid dimensions:")
    @printf("  - Latitudinal: %d points\n", size(u_cells)[2])
    @printf("  - Vertical   : %d levels\n", size(u_cells)[3])
    @printf("  - Time steps : %d\n", size(u_cells)[4])

    # Statistics by time
    println("\n" * "-"^80)
    println("Mid-Longitude Horizontal Velocity Statistics by Time Step (Sampled)")
    println("-"^80)
    @printf("%-15s %-15s %-15s %-15s\n", "Time (days)", "Max H-Speed", "Mean H-Speed", "P99")
    println("-"^80)

    max_speeds_mid = Float64[]
    sample_indices = round.(Int, range(1, size(speed_h, 4), length=min(20, size(speed_h, 4))))

    for t in axes(speed_h, 4)
        sh = speed_h[:, :, :, t]
        push!(max_speeds_mid, maximum(sh))
    end

    for t in sample_indices
        time_days = times[t]
        sh = speed_h[:, :, :, t]
        max_h_speed = maximum(sh)
        mean_h_speed = mean(sh)
        p99_speed = quantile(vec(sh), 0.99)

        @printf("%-15.4f %-15.6f %-15.6f %-15.6f\n",
                time_days, max_h_speed, mean_h_speed, p99_speed)
    end

    # Vertical profile analysis
    println("\n" * "-"^80)
    println("Vertical Velocity Profile (time-averaged)")
    println("-"^80)
    @printf("%-12s %-15s %-15s\n", "Depth Index", "Max Speed", "Mean Speed")
    println("-"^80)

    speed_h_tavg = mean(speed_h, dims=4)  # Average over time (dim 4): result is (1, 150, 135)

    # Sample depth levels
    sample_z = round.(Int, range(1, size(speed_h_tavg, 3), length=min(15, size(speed_h_tavg, 3))))
    for z_idx in sample_z
        s_at_depth = speed_h_tavg[1, :, z_idx]  # Get latitude profile at this depth
        max_s = maximum(s_at_depth)
        mean_s = mean(s_at_depth)
        @printf("%-12d %-15.6f %-15.6f\n", z_idx, max_s, mean_s)
    end

    # Find where largest velocities occur
    println("\n" * "-"^80)
    println("Peak Velocity Location in Mid-Longitude Slice")
    println("-"^80)

    max_idx = argmax(speed_h_tavg)
    y_idx, z_idx = max_idx[2], max_idx[3]  # dims 2 and 3 are lat and z
    max_speed_at_idx = speed_h_tavg[1, y_idx, z_idx]

    @printf("  - Latitude index : %d (of %d)\n", y_idx, size(speed_h_tavg, 2))
    @printf("  - Depth index    : %d (of %d)\n", z_idx, size(speed_h_tavg, 3))
    @printf("  - Max speed      : %.6f m/s\n", max_speed_at_idx)
end

function print_summary(max_speeds_all=nothing, times=nothing)
    """Print summary and recommendations."""
    println("\n" * "="^80)
    println(" ANALYSIS SUMMARY & RECOMMENDATIONS")
    println("="^80)

    if max_speeds_all !== nothing && times !== nothing
        # Check spike characteristics
        spike_idx = argmax(max_speeds_all)

        if spike_idx > 10  # Spike occurs after some runtime
            println("\n✓ Spike occurs AFTER model spinup (day > ~$(times[min(10, end)]))")

            # Check if velocity before spike is reasonable
            pre_spike_max = maximum(max_speeds_all[1:spike_idx-1])
            if pre_spike_max < 1.5
                println("✓ Pre-spike velocities are reasonable (< 1.5 m/s)")
            else
                println("⚠ Pre-spike velocities already elevated (> 1.5 m/s)")
            end

            # Recommendations
            println("\nDiagnostic suggestions:")
            if times[spike_idx] > 25 && times[spike_idx] < 35
                println("  1. Spike at day ~30 suggests monthly wind/restoring update issue")
                println("     → Check wind forcing interpolation at month boundaries")
                println("     → Verify sponge layer restoring timescale (currently 5 days)")
                println("     → Consider smoother wind field interpolation")
            end
            println("  2. Examine boundary conditions at spike time:")
            println("     → Check inflow/outflow balances at open boundaries")
            println("     → Verify wave radiation scheme (NormalRadiation vs PerturbationAdvection)")
            println("  3. Reduce CFL target if not already done:")
            println("     → Current: 0.6, try reducing to 0.4-0.5")
            println("  4. Check Laplacian/Biharmonic dissipation:")
            println("     → Consider reducing to bsose config (ν=10 m²/s)")
        end
    end

    println("\n" * "="^80)
end

function main()
    println("SOWinWaRP Velocity Diagnostics Tool")
    println("="^80)

    # Parse command-line arguments
    surface_file = length(ARGS) > 0 ? ARGS[1] : "model_surface_fields.nc"
    mid_lon_file = length(ARGS) > 1 ? ARGS[2] : "model_mid_lon.nc"

    # Load data
    ds_surface, ds_mid = load_datasets(surface_file, mid_lon_file)

    # Analyze
    result_surface = analyze_surface_velocities(ds_surface)
    analyze_mid_lon_velocities(ds_mid)

    # Print summary with recommendations
    if result_surface !== nothing
        max_speeds_all, times = result_surface
        print_summary(max_speeds_all, times)
    else
        print_summary()
    end

    # Close datasets
    ds_surface !== nothing && close(ds_surface)
    ds_mid !== nothing && close(ds_mid)

    println("\n✓ Analysis complete!")
    println("="^80)
end

main()
