using LinearAlgebra

"""
    dmd(X::Matrix{Float64}, dt::Float64=1.0; modes::Int=0, energy_threshold::Float64=0.9999)

Compute the Dynamic Mode Decomposition (DMD) of matrix X.

Arguments:
- `X`: Input data matrix where each column represents a snapshot
- `dt`: Time step between snapshots (default: 1.0)
- `modes`: Number of DMD modes to retain (default: 0, meaning use energy threshold)
- `energy_threshold`: Energy threshold for mode truncation (default: 0.9999)

Returns:
- `Φ`: DMD modes
- `λ`: DMD eigenvalues
- `b`: Mode amplitudes
- `ω`: DMD frequencies
- `retained_modes`: Number of modes retained
"""
function dmd(X::Matrix{Float64}, dt::Float64=1.0; modes::Int=0, energy_threshold::Float64=0.9999)
    # Split data into sequential snapshots
    X1 = X[:, 1:end-1]
    X2 = X[:, 2:end]
    
    # Compute SVD of X1
    U, Σ, V = svd(X1)
    
    # Calculate cumulative energy content
    energy = cumsum(Σ.^2) / sum(Σ.^2)
    
    # Determine number of modes to retain
    if modes > 0
        r = min(modes, length(Σ))
    else
        r = findfirst(energy .>= energy_threshold)
    end
    
    # Truncate SVD
    U_r = U[:, 1:r]
    Σ_r = Diagonal(Σ[1:r])
    V_r = V[:, 1:r]
    
    # Compute reduced matrix Ã
    Ã = U_r' * X2 * V_r * inv(Σ_r)
    
    # Eigendecomposition of Ã
    λ, W = eigen(Ã)
    
    # Compute DMD modes
    Φ = X2 * V_r * inv(Σ_r) * W
    
    # Compute mode amplitudes using QR factorization for maximum stability
    b = qr(Φ, ColumnNorm()) \ X1[:, 1]
    
    # Convert outputs to complex type for consistency
    Φ = convert(Matrix{ComplexF64}, Φ)
    λ = convert(Vector{ComplexF64}, λ)
    b = convert(Vector{ComplexF64}, b)
    
    # Compute DMD frequencies
    ω = log.(λ) / dt
    
    return Φ, λ, b, ω, r
end

"""
    reconstruct_dmd(Φ::Matrix{ComplexF64}, λ::Vector{ComplexF64}, b::Vector{ComplexF64}, 
                   t::Union{Vector{Float64}, AbstractRange{Float64}})

Reconstruct the dynamic data using DMD modes and eigenvalues.

Arguments:
- `Φ`: DMD modes
- `λ`: DMD eigenvalues
- `b`: Mode amplitudes
- `t`: Time points for reconstruction

Returns:
- `X_dmd`: Reconstructed data matrix
"""
function reconstruct_dmd(Φ::Matrix{ComplexF64}, λ::Vector{ComplexF64}, 
                        b::Vector{ComplexF64}, t::Union{Vector{Float64}, AbstractRange{Float64}})
    # Initialize reconstruction matrix
    X_dmd = zeros(ComplexF64, size(Φ, 1), length(t))
    
    # Reconstruct data
    for (i, ti) in enumerate(t)
        X_dmd[:, i] = Φ * (b .* λ.^ti)
    end
    
    # Return real part if reconstruction is approximately real
    if maximum(abs.(imag.(X_dmd))) < 1e-10
        return real(X_dmd)
    else
        return X_dmd
    end
end 