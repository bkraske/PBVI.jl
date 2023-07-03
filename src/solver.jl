Base.@kwdef struct PBVISolver{INIT,RNG<:AbstractRNG} <: Solver
    ϵ::Float64          = 1e-2
    max_time::Float64   = Inf
    max_iter::Int       = 10
    init::INIT          = BlindLowerBound()
    rng::RNG            = default_rng(1337)
    verbose::Bool       = false
    witness_b::Bool     = false
end

function POMDPs.solve(sol::PBVISolver, pomdp::POMDP)
    t0 = time()
    max_iter_len = length(string(sol.max_iter))
    pad_size = max(4, max_iter_len)
    tree = PBVITree(pomdp)
    push!(tree.b, tree.pomdp.initialstate)
    push!(tree.is_terminal, is_terminal_belief(tree.pomdp.initialstate, tree.pomdp.isterminal))
    push!(tree.b_children,NO_CHILDREN)
    expand!(tree, 1)

    init_alpha = solve(sol.init, tree.pomdp)
    for (α,a) ∈ alphapairs(init_alpha)
        push!(tree.Γ, AlphaVec(α, a, 0))
    end
    v_root = belief_value(tree.Γ, tree.b[1])
    ϵ = Inf

    iter = 0
    if sol.verbose
        init_str = rpad("iter", pad_size)*" | " * "v_root" * " | " * "|Γ|"
        println('\n', init_str)
        println('-'^length(init_str))
        println(rpad("$iter", pad_size), " | ", round(v_root, sigdigits=3), " | ", length(tree.Γ))
    end
    while (time() - t0 < sol.max_time) && (iter < sol.max_iter)
        iter += 1
        expand!(sol, tree)
        backup_while_diff!(sol, tree)

        v_root_new = belief_value(tree.Γ, tree.b[1])
        ϵ = abs(v_root_new - v_root)
        v_root = v_root_new
        sol.verbose && println(
            rpad("$iter", pad_size), " | ", round(v_root, sigdigits=3), " | ", length(tree.Γ)
        )
    end

    if sol.witness_b
        return AlphaVectorPolicy(
            pomdp,
            getproperty.(tree.Γ, :v),
            ordered_actions(pomdp)[getproperty.(tree.Γ, :a)]
        ), tree.b[getproperty.(tree.Γ, :b)], tree.depth[getproperty.(tree.Γ, :b)]
    else
        return AlphaVectorPolicy(
            pomdp,
            getproperty.(tree.Γ, :v),
            ordered_actions(pomdp)[getproperty.(tree.Γ, :a)]
        )
    end
end

function backup_while_diff!(sol, tree)
    ϵ = Inf
    root_val = belief_value(tree.Γ, tree.b[1])
    while ϵ > sol.ϵ
        backup!(tree)
        new_root_val = belief_value(tree.Γ, tree.b[1])
        ϵ = new_root_val - root_val
        root_val = new_root_val
    end
    return root_val
end

function expand!(sol::PBVISolver, tree::PBVITree)
    (;rng) = sol
    (;pomdp) = tree
    A = actions(pomdp)
    O = observations(pomdp)
    old_real = copy(tree.real)

    for b_idx ∈ old_real
        b_children = tree.b_children[b_idx]
        tree.is_terminal[b_idx] && continue
        d_max = 0.0
        bp_idx_max = 0
        for a ∈ A
            ba_idx = b_children[a]
            o = rand(rng, O)
            bp_idx = tree.ba_children[ba_idx][o]
            bp = tree.b[bp_idx]
            d = distance_to_tree(tree, bp)
            if d > d_max
                d_max = d
                bp_idx_max = bp_idx
            end
        end
        d_max > 0. && expand!(tree, bp_idx_max)
    end
end
