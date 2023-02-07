Base.@kwdef struct PBVISolver{INIT} <: Solver
    ϵ::Float64          = 1e-2
    max_time::Float64   = Inf
    max_iter::Int       = 10
    init::INIT          = BlindLowerBound()
end

function POMDPs.solve(sol::PBVISolver, pomdp::POMDP)
    t0 = time()
    tree = PBVITree(pomdp)
    push!(tree.b, tree.pomdp.initialstate)
    push!(tree.is_terminal, is_terminal_belief(tree.pomdp.initialstate, tree.pomdp.isterminal))
    init_alpha = solve(sol.init, tree.pomdp)
    for (α,a) ∈ alphapairs(init_alpha)
        push!(tree.Γ, AlphaVec(α, a))
    end
    fringe_begin = expand!(tree, 1)
    v_root = belief_value(tree.Γ, tree.b[1])
    ϵ = Inf

    iter = 0
    # while (ϵ > sol.ϵ) && (time() - t0 < sol.max_time) && (iter < sol.max_iter)
    while (time() - t0 < sol.max_time) && (iter < sol.max_iter)
        iter += 1
        fringe_begin = expand!(tree, fringe_begin)
        backup!(tree, fringe_begin)

        v_root_new = belief_value(tree.Γ, tree.b[1])
        ϵ = abs(v_root_new - v_root)
        v_root = v_root_new
    end

    return AlphaVectorPolicy(
        pomdp,
        getproperty.(tree.Γ, :v),
        ordered_actions(pomdp)[getproperty.(tree.Γ, :a)]
    )
end

function expand!(tree::PBVITree, fringe_begin)
    pomdp = tree.pomdp
    A = actions(pomdp)
    O = observations(pomdp)

    fringe_end = length(tree.b)

    for b_idx ∈ fringe_begin:fringe_end
        if !tree.is_terminal[b_idx]
            b = tree.b[b_idx]
            n_b = length(tree.b_children)
            b_children = n_b+1 : n_b+length(A)
            for a ∈ A
                n_ba = length(tree.ba_children)
                ba_children = n_ba+1 : n_ba+length(O)
                pred = dropzeros!(mul!(tree.cache.pred, pomdp.T[a],b))
                for o ∈ O
                    bp = corrector(pomdp, pred, a, o)
                    po = sum(bp)
                    po > 0. && (bp.nzval ./= po)

                    terminal = iszero(po) || is_terminal_belief(bp, pomdp.isterminal)

                    push!(tree.b, bp)
                    push!(tree.is_terminal, terminal)
                end
                push!(tree.ba_children, ba_children)
            end
            push!(tree.b_children, b_children)
        end
    end
    return fringe_end + 1
end
