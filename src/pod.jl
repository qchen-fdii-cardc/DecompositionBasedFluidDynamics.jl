using LinearAlgebra  # Add this at the top of the file

"""
    pod(X::Matrix{Float64}; modes::Int=0, energy_threshold::Float64=0.9999)

Compute the Proper Orthogonal Decomposition (POD) of matrix X.

Arguments:
- `X`: Input data matrix where each column represents a snapshot
- `modes`: Number of POD modes to retain (default: 0, meaning use energy threshold)
- `energy_threshold`: Energy threshold for mode truncation (default: 0.9999)

Returns:
- `U`: POD modes (spatial basis vectors)
- `S`: Singular values
- `V`: Temporal coefficients (modes in rows, time samples in columns)
- `retained_modes`: Number of modes retained
"""
function pod(X::Matrix{Float64}; modes::Int=0, energy_threshold::Float64=0.9999)
    # Compute SVD of the data matrix
    U, S, V = svd(X)
    
    # Calculate cumulative energy content
    energy = cumsum(S.^2) / sum(S.^2)
    
    # Determine number of modes to retain
    if modes > 0
        retained_modes = min(modes, length(S))
    else
        retained_modes = findfirst(energy .>= energy_threshold)
    end
    
    # Truncate the decomposition
    U = U[:, 1:retained_modes]
    S = S[1:retained_modes]
    V = Matrix(V[:, 1:retained_modes]')  # Convert Adjoint to Matrix
    
    return U, S, V, retained_modes
end

"""
    reconstruct(U::Matrix{Float64}, S::Vector{Float64}, V::Matrix{Float64})

Reconstruct the original data matrix from POD modes and coefficients.

Returns:
- `X_reconstructed`: Reconstructed data matrix
"""
function reconstruct(U::Matrix{Float64}, S::Vector{Float64}, V::Matrix{Float64})
    return U * Diagonal(S) * V
end
