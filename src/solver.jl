Base.@kwdef struct PBVISolver{INIT} <: Solver
    ϵ::Float64          = 1e-2
    max_time::Float64   = Inf
    max_iter::Int       = 10
    init::INIT          = BlindLowerBound()
    verbose::Bool       = false
end

function POMDPs.solve(sol::PBVISolver, pomdp::POMDP)
    t0 = time()
    max_iter_len = length(string(sol.max_iter))
    tree = PBVITree(pomdp)
    push!(tree.b, tree.pomdp.initialstate)
    push!(tree.is_terminal, is_terminal_belief(tree.pomdp.initialstate, tree.pomdp.isterminal))
    push!(tree.b_children,NO_CHILDREN)
    expand!(tree, 1)

    init_alpha = solve(sol.init, tree.pomdp)
    for (α,a) ∈ alphapairs(init_alpha)
        push!(tree.Γ, AlphaVec(α, a))
    end
    v_root = belief_value(tree.Γ, tree.b[1])
    ϵ = Inf

    iter = 0
    if sol.verbose
        iter_str = rpad("iter: $iter", 6+max_iter_len)
        v_str = "v_root: $(round(v_root, sigdigits=3))"
        println(iter_str, " | ", v_str)
    end
    while (time() - t0 < sol.max_time) && (iter < sol.max_iter)
        iter += 1
        expand!(tree)
        backup!(tree)

        v_root_new = belief_value(tree.Γ, tree.b[1])
        ϵ = abs(v_root_new - v_root)
        v_root = v_root_new
        if sol.verbose
            iter_str = rpad("iter: $iter", 6+max_iter_len)
            v_str = "v_root: $(round(v_root, sigdigits=3))"
            println(iter_str, " | ", v_str)
        end
    end

    return AlphaVectorPolicy(
        pomdp,
        getproperty.(tree.Γ, :v),
        ordered_actions(pomdp)[getproperty.(tree.Γ, :a)]
    )
end

function expand!(tree::PBVITree)
    pomdp = tree.pomdp
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
            o = rand(O)
            bp_idx = tree.ba_children[ba_idx][o]
            bp = tree.b[bp_idx]
            d = distance_to_tree(tree, bp)
            if d > d_max
                d_max = d
                bp_idx_max = bp_idx
            end
        end
        expand!(tree, bp_idx_max)
    end
end
