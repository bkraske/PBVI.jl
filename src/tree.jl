struct PBVITree
    b::Vector{Vector{Float64}}
    b_children::Vector{UnitRange{Int}}
    ba_children::Vector{UnitRange{Int}}
    V::Vector{Float64}
    Qa::Vector{Vector{Float64}}
    Γ::Vector{AlphaVec}
end

function PBVITree()
    return PBVITree(
        Vector{Float64}[],
        UnitRange{Int}[],
        UnitRange{Int}[],
        Float64[],
        Vector{Float64}[],
        AlphaVec[]
    )
end
