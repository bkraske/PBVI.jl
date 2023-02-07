struct TreeCache
    pred::SparseVector{Float64,Int}
    alpha::Vector{Float64}
    Γ::Array{Float64,3}
    Γ_tmp::Vector{AlphaVec}
    Oᵀ::Vector{SparseMatrixCSC{Float64, Int64}}
end

function TreeCache(pomdp::ModifiedSparseTabular)
    Ns = n_states(pomdp)
    Na = n_actions(pomdp)
    No = n_observations(pomdp)

    pred = Vector{Float64}(undef, Ns)
    alpha = Vector{Float64}(undef, Ns)
    Γ = Array{Float64,3}(undef, Ns, No, Na)
    Γ_tmp = AlphaVec[]
    Oᵀ = map(sparse ∘ transpose, pomdp.O)
    return TreeCache(pred, alpha, Γ, Γ_tmp, Oᵀ)
end
