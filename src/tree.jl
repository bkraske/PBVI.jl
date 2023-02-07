struct PBVITree
    b::Vector{Vector{Float64}}
    b_children::Vector{UnitRange{Int}}
    is_terminal::BitVector
    ba_children::Vector{UnitRange{Int}}
    V::Vector{Float64}
    Qa::Vector{Vector{Float64}}
    cache::TreeCache
    Γ::Vector{AlphaVec}
    pomdp::ModifiedSparseTabular
end

function PBVITree(pomdp::ModifiedSparseTabular)
    return PBVITree(
        Vector{Float64}[],
        UnitRange{Int}[],
        BitVector(),
        UnitRange{Int}[],
        Float64[],
        Vector{Float64}[],
        TreeCache(pomdp),
        AlphaVec[],
        pomdp
    )
end

PBVITree(pomdp::POMDP) = PBVITree(ModifiedSparseTabular(pomdp))
