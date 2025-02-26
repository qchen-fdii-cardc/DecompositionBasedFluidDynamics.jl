using Test
using LinearAlgebra
using DecompositionBasedFluidDynamics
using Plots
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
