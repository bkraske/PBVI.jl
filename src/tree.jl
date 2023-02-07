const Belief = SparseVector{Float64,Int}

Base.@kwdef struct PBVITree
    b::Vector{Belief}                   = Belief[]
    real::Vector{Int}                   = Int[]
    b_children::Vector{UnitRange{Int}}  = UnitRange{Int}[]
    is_terminal::BitVector              = BitVector()
    ba_children::Vector{UnitRange{Int}} = UnitRange{Int}[]
    V::Vector{Float64}                  = Float64[]
    Qa::Vector{Vector{Float64}}         = Vector{Float64}[]
    Γ::Vector{AlphaVec}                 = AlphaVec[]
    cache::TreeCache
    pomdp::ModifiedSparseTabular
end

function PBVITree(pomdp::ModifiedSparseTabular)
    return PBVITree(;
        cache = TreeCache(pomdp),
        pomdp
    )
end

PBVITree(pomdp::POMDP) = PBVITree(ModifiedSparseTabular(pomdp))

const NO_CHILDREN = 1:0

"""
If empty and terminal, there's nothing we can really say about it
If empty and nonterminal, then it's not real
If not empty and nonterminal, then it's real
"""
@inline is_real(tree, b_idx) = b_idx ∈ tree.real



"""
explore b' = τ(bao)
b has already been explored
-> b' is already in the tree
-> bp children empty but exists
-> b

----
if terminal, still want to add to real, but can't expand
if real, already expanded, don't want to add to real and don't want to expand
if not real and not terminal, want to add to real and expand
"""
function expand!(tree, b_idx::Int)
    if !is_real(tree, b_idx)
        push!(tree.real, b_idx)
        if !tree.is_terminal[b_idx]
            push!(tree.real, b_idx)
            expand_belief!(tree, b_idx)
        end
    end
end

function expand_belief!(tree, b_idx::Int)
    (;pomdp) = tree
    b = tree.b[b_idx]
    A = actions(pomdp)
    O = observations(pomdp)
    n_ba = length(tree.ba_children)
    tree.b_children[b_idx] = (n_ba+1) : (n_ba + length(A))
    for a ∈ A
        n_b = length(tree.b)
        push!(tree.ba_children, (n_b+1) : (n_b+length(O)) )
        pred = dropzeros!(pomdp.T[a]*b)
        for o ∈ O
            bp = corrector(pomdp, pred, a, o)
            po = sum(bp)
            po > 0. && (bp.nzval ./= po)
            terminal = iszero(po) || is_terminal_belief(bp, pomdp.isterminal)
            push!(tree.b, bp)
            push!(tree.is_terminal, terminal)
            push!(tree.b_children, NO_CHILDREN)
        end
    end
end

function distance_to_tree(tree, b′)
    d_min = Inf
    b_cache = similar(b′)
    for b in tree.real
        d = norm(b_cache .= b .- b′, 1)
        d < d_min && (d_min = d)
    end
    return d_min
end
