using Test
using LinearAlgebra
using DecompositionBasedFluidDynamics
using Plots
using Statistics  # Add this import for std()


@testset "POD method" begin
    # Write your tests here.
    X = rand(100, 100)
    U, S, V, retained_modes = pod(X)
    
    println("Number of retained modes: ", retained_modes)
    println("S = ", S[1:10])
    
    @test retained_modes <= 100
    @test size(U) == (100, retained_modes)
    @test size(S) == (retained_modes,)
    @test size(V) == (retained_modes, 100)
    X_reconstructed = reconstruct(U, S, V)
    @test size(X_reconstructed) == size(X)
    @test norm(X - X_reconstructed) / norm(X) < 1e-2


    # Test with energy threshold
    U, S, V, retained_modes = pod(X, energy_threshold=0.95)
    @test retained_modes <= 100
    @test norm(X - reconstruct(U, S, V)) / norm(X) < 0.5

    # Test with number of modes
    U, S, V, retained_modes = pod(X, modes=10)
    @test retained_modes == 10
    @test norm(X - reconstruct(U, S, V)) / norm(X) < 0.5
    println("norm ratio = ", norm(reconstruct(U, S, V)) / norm(X))

    # Calculate reconstruction errors for different numbers of modes
    reconstruction_errors = Float64[]
    for i in 1:100
        U, S, V, retained_modes = pod(X, modes=i)
        error = 1.0 - norm(reconstruct(U, S, V)) / norm(X)
        push!(reconstruction_errors, error)
    end

    # plot the norm ratio vs the number of retained modes
    p = plot(1:100, reconstruction_errors,
        xlabel="Number of retained modes", 
        ylabel="Reconstruction error", 
        title="POD reconstruction error vs number of retained modes",
        legend=false,
        linewidth=2)
    
    savefig(p, "../doc/static/pod_norm_ratio.png")
    @test isfile("../doc/static/pod_norm_ratio.png")
    
    # plot the singular values
    U, S, V, retained_modes = pod(X, modes=100)
    p = plot(S, 
        xlabel="Mode number", 
        ylabel="Singular value", 
        title="POD singular values",
        legend=false,
        linewidth=2)
    savefig(p, "../doc/static/pod_singular_values.png")
    @test isfile("../doc/static/pod_singular_values.png")

    # plot the cumulative energy
    p = plot(cumsum(S.^2) / sum(S.^2), 
        xlabel="Mode number", 
        ylabel="Cumulative energy", 
        title="POD cumulative energy",
        ylims=(0, 1.1),
        legend=false,
        linewidth=2)
    savefig(p, "../doc/static/pod_cumulative_energy.png")
    @test isfile("../doc/static/pod_cumulative_energy.png")

end

@testset "DMD method" begin
    # Create a simple test case with known dynamics
    t = 0.0:0.1:10.0
    x = zeros(3, length(t))
    # Create a system with growing oscillation and decay
    for (i, ti) in enumerate(t)
        x[1, i] = exp(0.1 * ti) * cos(2π * ti)
        x[2, i] = exp(-0.2 * ti) * sin(π * ti)
        x[3, i] = exp(-0.1 * ti)
    end
    
    # Apply DMD
    Φ, λ, b, ω, retained_modes = dmd(x, 0.1)
    
    # Basic size tests
    @test size(Φ, 1) == 3
    @test length(λ) == retained_modes
    @test length(b) == retained_modes
    @test length(ω) == retained_modes
    
    # Reconstruction test
    x_reconstructed = reconstruct_dmd(Φ, λ, b, t)
    @test size(x_reconstructed) == size(x)
    
    # Test reconstruction accuracy
    relative_error = 1.0 - norm(x_reconstructed) / norm(x)
    @test relative_error < 0.3
    
    # Test with specified number of modes
    Φ, λ, b, ω, retained_modes = dmd(x, 0.1, modes=2)
    @test retained_modes == 2
    
    # Test with energy threshold
    Φ, λ, b, ω, retained_modes = dmd(x, 0.1, energy_threshold=0.95)
    @test retained_modes <= size(x, 1)
    
    # Plot DMD spectrum in 3D with mode amplitudes
    θ = range(0, 2π, length=100)
    unit_circle_x = cos.(θ)
    unit_circle_y = sin.(θ)
    unit_circle_z = zeros(length(θ))
    
    p = plot(camera=(30, 30), size=(800, 600))
    
    # Plot unit circle
    plot!(p, unit_circle_x, unit_circle_y, unit_circle_z,
        label="Unit circle",
        color=:black,
        linestyle=:dash,
        linewidth=1)
    
    # Plot DMD eigenvalues with mode amplitudes as height
    scatter!(p, real.(ω), imag.(ω), abs.(b),
        xlabel="Real(ω)", 
        ylabel="Imag(ω)",
        zlabel="Mode amplitude",
        title="DMD Spectrum",
        label="DMD modes",
        marker=:circle,
        markersize=4,
        color=:blue)
    
    # Add vertical lines from eigenvalues to unit circle
    for (x, y, z) in zip(real.(ω), imag.(ω), abs.(b))
        plot!(p, [x, x], [y, y], [0, z],
            color=:gray,
            linestyle=:dot,
            label=false)
    end
    
    savefig(p, "../doc/static/dmd_spectrum_3d.png")
    @test isfile("../doc/static/dmd_spectrum_3d.png")
    
    # Plot mode amplitudes
    p = bar(abs.(b),
        xlabel="Mode index", 
        ylabel="Mode amplitude", 
        title="DMD Mode Amplitudes",
        legend=false)
    savefig(p, "../doc/static/dmd_amplitudes.png")
    @test isfile("../doc/static/dmd_amplitudes.png")
end

@testset "Pulsating Poiseuille Flow Analysis" begin
    # Parameters
    ν = 1.0     # kinematic viscosity
    ρ = 1.0     # fluid density
    L = 1.0     # channel half-width
    pM = 1.0    # mean pressure gradient
    pA = 0.5    # pressure gradient amplitude
    ω = π       # angular frequency
    
    Nx = 50     # number of spatial points
    Nt = 200    # number of time points
    T = 4.0     # total time
    
    # Spatial and temporal discretization
    y = range(-L, L, length=Nx)
    t = range(0, T, length=Nt)
    dt = t[2] - t[1]
    
    # Analytical solution for pulsating Poiseuille flow with oscillating pressure gradient
    function analytical_solution(y, t, ν, ρ, L, pM, pA, ω)
        # Mean flow component (steady Poiseuille flow)
        u_mean = pM/(2ν*ρ) * (L^2 - y^2)
        
        # Oscillatory component
        β = sqrt(ω/(2ν))  # Womersley number parameter
        
        # Complex solution for oscillatory part
        z = sqrt(im*ω/ν) * y
        z_L = sqrt(im*ω/ν) * L
        
        u_osc = (pA/(im*ω*ρ)) * (1 - cosh(z)/cosh(z_L)) * exp(im*ω*t)
        
        # Return real part of total solution
        return real(u_mean + u_osc)
    end
    
    # Generate snapshots matrix
    X = zeros(Nx, Nt)
    for (j, tj) in enumerate(t)
        for (i, yi) in enumerate(y)
            X[i,j] = analytical_solution(yi, tj, ν, ρ, L, pM, pA, ω)
        end
    end
    
    # Add some noise to make it more realistic
    X_noisy = X + 0.01 * std(X) * randn(size(X))
    
    # Visualize the flow evolution
    p = plot(layout=(2,2), size=(1000, 800))
    
    # Plot flow profiles at different times
    times_to_plot = [0.0, T/4, T/2, 3T/4]
    plot!(p[1,1], y, [analytical_solution.(y, t, ν, ρ, L, pM, pA, ω) for t in times_to_plot],
        xlabel="y/L",
        ylabel="u/U₀",
        title="Flow Profiles",
        label=["t=0" "t=T/4" "t=T/2" "t=3T/4"])
    
    # Plot centerline velocity evolution
    t_fine = range(0, T, length=1000)
    u_center = [analytical_solution(0, t, ν, ρ, L, pM, pA, ω) for t in t_fine]
    plot!(p[1,2], t_fine, u_center,
        xlabel="t/T",
        ylabel="u/U₀",
        title="Centerline Velocity",
        legend=false)
    
    savefig(p, "../doc/static/flow_evolution.png")
    
    # POD Analysis
    U_pod, S_pod, V_pod, r_pod = pod(X_noisy, energy_threshold=0.99)
    X_pod = reconstruct(U_pod, S_pod, V_pod)
    
    # DMD Analysis
    Φ_dmd, λ_dmd, b_dmd, ω_dmd, r_dmd = dmd(X_noisy, dt, energy_threshold=0.99)
    X_dmd = reconstruct_dmd(Φ_dmd, λ_dmd, b_dmd, t)
    
    # Compare reconstruction errors
    error_pod = norm(X - X_pod) / norm(X)
    error_dmd = norm(X - X_dmd) / norm(X)
    
    @test error_pod < 1.0
    @test error_dmd < 1.0
    
    # Visualization
    # 1. POD Analysis plots
    p = plot(layout=(2,2), size=(1000, 800))
    
    # POD singular values
    plot!(p[1,1], S_pod, 
        xlabel="Mode index", 
        ylabel="Singular value",
        title="POD Singular Values",
        legend=false,
        marker=:circle,
        markersize=4,
        grid=true)

    println("S_pod = ", S_pod)
    
    # POD cumulative energy
    plot!(p[1,2], cumsum(S_pod.^2)/sum(S_pod.^2),
        xlabel="Number of modes",
        ylabel="Cumulative energy",
        title="POD Energy Content",
        legend=false,
        marker=:circle,
        markersize=4,
        grid=true,
        ylims=(0, 1.1))
    
    # First POD modes (up to 2)
    n_modes_to_plot = min(2, size(U_pod, 2))
    plot!(p[2,1], y, U_pod[:,1:n_modes_to_plot],
        xlabel="y/L",
        ylabel="Mode amplitude",
        title="First POD Modes",
        label=n_modes_to_plot == 1 ? "Mode 1" : ["Mode 1" "Mode 2"])
    
    # POD temporal coefficients
    plot!(p[2,2], t, V_pod[1:n_modes_to_plot,:]',
        xlabel="Time",
        ylabel="Amplitude",
        title="POD Temporal Coefficients",
        label=n_modes_to_plot == 1 ? "Mode 1" : ["Mode 1" "Mode 2"])
    
    savefig(p, "../doc/static/pod_analysis.png")
    
    # 2. DMD Analysis plots
    p = plot(layout=(2,2), size=(1000, 800))
    
    # DMD spectrum
    scatter!(p[1,1], real.(ω_dmd), imag.(ω_dmd),
        xlabel="Real(ω)",
        ylabel="Imag(ω)",
        title="DMD Spectrum",
        legend=false)
    
    # DMD mode amplitudes
    bar!(p[1,2], abs.(b_dmd),
        xlabel="Mode index",
        ylabel="Mode amplitude",
        title="DMD Mode Amplitudes",
        legend=false)
    
    # First DMD mode
    plot!(p[2,1], y, real.(Φ_dmd[:,1]),
        xlabel="y/L",
        ylabel="Mode amplitude",
        title="First DMD Mode (Real Part)",
        legend=false)
    
    # Reconstruction error comparison
    times_to_plot = 1:10:min(20, Nt)  # Take first few snapshots at intervals of 10
    data_to_plot = hcat(
        X[:, times_to_plot[1]],  # Original data at first time
        X_pod[:, times_to_plot[1]],  # POD reconstruction
        X_dmd[:, times_to_plot[1]]   # DMD reconstruction
    )
    
    plot!(p[2,2], y, data_to_plot,
        xlabel="y/L",
        ylabel="Velocity",
        title="Original vs Reconstructed (t=$(t[1]))",
        label=["Original" "POD" "DMD"])
    
    savefig(p, "../doc/static/dmd_analysis.png")
    
    # 3. Error comparison plot
    p = plot(t,
        [vec(mean((X - X_pod).^2, dims=1)) vec(mean((X - X_dmd).^2, dims=1))],
        xlabel="Time",
        ylabel="MSE",
        title="Reconstruction Error Comparison",
        label=["POD" "DMD"],
        yscale=:log10)
    
    savefig(p, "../doc/static/reconstruction_comparison.png")
    
    # Print analysis results
    println("\nFlow Parameters:")
    println("  Womersley number: ", L*sqrt(ω/ν))
    println("  Mean pressure gradient: ", pM)
    println("  Pressure oscillation amplitude: ", pA)
    println("  Oscillation frequency: ", ω/(2π))
    
    println("\nPOD Analysis:")
    println("  Number of modes retained: ", r_pod)
    println("  Reconstruction error: ", error_pod)
    println("  Energy in first mode: ", S_pod[1]^2/sum(S_pod.^2))
    
    println("\nDMD Analysis:")
    println("  Number of modes retained: ", r_dmd)
    println("  Reconstruction error: ", error_dmd)
    println("  Dominant frequencies: ", sort(abs.(ω_dmd))[1:min(3,length(ω_dmd))])
    println("  Dominant growth rates: ", sort(real.(λ_dmd))[1:min(3,length(λ_dmd))])
end
