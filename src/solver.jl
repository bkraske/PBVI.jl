#Borrowing from github.com/JuliaPOMDP/PointBasedValueIteration.jl

struct PBVISolver2 <: Solver
    max_iterations::Int64
    ϵ::Float64
    verbose::Bool
end

function PBVISolver2(; max_iterations::Int64=10, ϵ::Float64=0.01, verbose::Bool=false)
    return PBVISolver2(max_iterations, ϵ, verbose)
end

"""
    AlphaVec
Pair of alpha vector and corresponding action.
# Fields
- `alpha` α vector
- `action` action corresponding to α vector
"""
struct AlphaVec
    alpha::Vector{Float64}
    action::Any
end

function max_alpha_val(Γ, b)
    max_α = first(Γ)
    max_val = -Inf
    for α ∈ Γ
        val = dot(α, b)
        if val > max_val
            max_α = α
            max_val = val
        end
    end
    return max_α
end

function max_alpha_val_ind(Γ, b)
    max_ind = 1
    max_val = -Inf
    for (i,α) ∈ enumerate(Γ)
        val = dot(α, b)
        if val > max_val
            max_ind = i
            max_val = val
        end
    end
    return max_ind
end

# adds probabilities of terminals in b to b′ and normalizes b′
function belief_norm(pomdp, b, b′, terminals, not_terminals)
    if sum(view(b′,not_terminals)) != 0.0
        if !isempty(terminals)
            @views b′[not_terminals] ./= (sum(b′[not_terminals]) / (1. - sum(b[terminals]) - sum(b′[terminals])))
            @views b′[terminals] += b[terminals]
        else
            @views b′[not_terminals] ./= sum(b′[not_terminals])
        end
    else
        @views b′[terminals] += b[terminals]
        @views b′[terminals] ./= sum(b′[terminals])
    end
    return b′
end

# Backups belief with α vector maximizing dot product of itself with belief b
function backup_belief(pomdp::POMDP, Γ, b, spomdp)
    S = ordered_states(pomdp)
    A = ordered_actions(pomdp)
    # O = ordered_observations(pomdp)
    γ = discount(pomdp)
    # r = StateActionReward(pomdp)

    Γa = Vector{Vector{Float64}}(undef, length(A))

    terminals = spomdp.terminal_states
    not_terminals = deleteat!(collect(1:length(S)),terminals)
    # not_terminals = [stateindex(pomdp, s) for s in S if !isterminal(pomdp, s)]
    # terminals = [stateindex(pomdp, s) for s in S if isterminal(pomdp, s)]
    for a in A
        a_ind = actionindex(pomdp,a)
        O1 = spomdp.O[a_ind]
        T = spomdp.T[a_ind]
        O_len = length(O1[1,:])
        Γao = Vector{Vector{Float64}}(undef,O_len)
        # @show [pdf(transition(pomdp, S[is], a), sp) * b.b[is] for sp in S, is in not_terminals]
        # @show trans_probs = dropdims(sum([pdf(transition(pomdp, S[is], a), sp) * b.b[is] for sp in S, is in not_terminals], dims=2), dims=2)
        trans_probs = zeros(length(S))
        for is ∈ not_terminals
            for sp ∈ S
                sp_i = stateindex(pomdp,sp)
                trans_probs[sp_i] += b.b[is]*T[is,sp_i]
            end
        end
        # @show trans_probs

        if !isempty(terminals)
            trans_probs[terminals] .+= b.b[terminals]
        end

        for o_i in 1:O_len
            # update beliefs
            # obs_probs = pdf.(map(sp -> observation(pomdp, a, sp), S), [o])
            # @show pdf.(map(sp -> observation(pomdp, a, sp), S), [ordered_observations(pomdp)[o_i]])
            obs_probs = O1[:,o_i]
            # @show obs_probs
            b′ = Vector(obs_probs .* trans_probs)

            if sum(b′) > 0.0
                b′ = DiscreteBelief(pomdp, b.state_list, belief_norm(pomdp, b.b, b′, terminals, not_terminals))
            else
                b′ = DiscreteBelief(pomdp, b.state_list, zeros(length(S)))
            end

            # extract optimal alpha vector at resulting belief
            Γao[o_i] = max_alpha_val(Γ, b′.b)
        end

        # construct new alpha vectors
        # Γa[actionindex(pomdp, a)] = [r(s, a) + (!isterminal(pomdp, s) ? (γ * sum(pdf(transition(pomdp, s, a), sp) * pdf(observation(pomdp, s, a, sp), o) * Γao[i][j]
        #                                                                          for (j, sp) in enumerate(S), (i, o) in enumerate(O))) : 0.0)
        #                              for s in S]
        
        Γa[a_ind] = zeros(length(S))
        for s_i in 1:length(S)
            Γa[a_ind][s_i] = spomdp.R[s_i,a_ind] 
            for sp_i in 1:length(S)
                for o_i in 1:O_len
                    if s_i ∉ terminals
                        Γa[a_ind][s_i] += γ*T[s_i,sp_i]*O1[sp_i,o_i]*Γao[o_i][sp_i]
                    end
                end
            end
        end
    end

    # find the optimal alpha vector
    idx = max_alpha_val_ind(Γa, b.b)
    alphavec = AlphaVec(Γa[idx], A[idx])

    return alphavec
end

# Iteratively improves α vectors until the gap between steps is lesser than ϵ
function improve(pomdp, B, Γ, solver)
    alphavecs = nothing
    spomdp = SparseTabularPOMDP(pomdp)
    while true
        Γold = Γ
        alphavecs = [backup_belief(pomdp, Γold, b, spomdp) for b in B]
        Γ = [alphavec.alpha for alphavec in alphavecs]
        prec = max([sum(abs.(dot(α1, b.b) .- dot(α2, b.b))) for (α1, α2, b) in zip(Γold, Γ, B)]...)
        if solver.verbose
            println("    Improving alphas, maximum gap between old and new α vector: $(prec)")
        end
        prec > solver.ϵ || break
    end

    return Γ, alphavecs
end

# Returns all possible, not yet visited successors of current belief b
function successors(pomdp, b, Bs)
    S = ordered_states(pomdp)
    not_terminals = [stateindex(pomdp, s) for s in S if !isterminal(pomdp, s)]
    terminals = [stateindex(pomdp, s) for s in S if isterminal(pomdp, s)]
    succs = []

    for a in actions(pomdp)
        trans_probs = dropdims(sum([pdf(transition(pomdp, S[is], a), sp) * b[is] for sp in S, is in not_terminals], dims=2), dims=2)
        if !isempty(terminals)
            trans_probs[terminals] .+= b[terminals]
        end

        for o in observations(pomdp)
            #update belief
            obs_probs = pdf.(map(sp -> observation(pomdp, a, sp), S), [o])
            b′ = obs_probs .* trans_probs


            if sum(b′) > 0.0
                b′ = belief_norm(pomdp, b, b′, terminals, not_terminals)

                if !in(b′, Bs)
                    push!(succs, b′)
                end
            end
        end
    end

    return succs
end

# Computes distance of successor to the belief vectors in belief space
function succ_dist(pomdp, bp, B)
    dist = [norm(bp - b.b, 1) for b in B]
    return max(dist...)
end

# Expands the belief space with the most distant belief vector
# Returns new belief space, set of belifs and early termination flag
function expand(pomdp, B, Bs)
    B_new = copy(B)
    for b in B
        succs = successors(pomdp, b.b, Bs)
        if length(succs) > 0
            b′ = succs[argmax([succ_dist(pomdp, bp, B) for bp in succs])]
            push!(B_new, DiscreteBelief(pomdp, b′))
            push!(Bs, b′)
        end
    end

    return B_new, Bs, length(B) == length(B_new)
end

# 1: B ← {b0}
# 2: while V has not converged to V∗ do
# 3:    Improve(V, B)
# 4:    B ← Expand(B)
function solve(solver::PBVISolver2, pomdp::POMDP)
    S = ordered_states(pomdp)
    A = ordered_actions(pomdp)
    γ = discount(pomdp)
    r = StateActionReward(pomdp)

    # best action worst state lower bound
    α_init = 1 / (1 - γ) * maximum(minimum(r(s, a) for s in S) for a in A)
    Γ = [fill(α_init, length(S)) for a in A]

    #init belief, if given distribution, convert to vector
    init = initialize_belief(DiscreteUpdater(pomdp), initialstate(pomdp))
    B = [init]
    Bs = Set([init.b])

    if solver.verbose
        println("Running PBVI solver on $(typeof(pomdp)) problem with following settings:\n    max_iterations = $(solver.max_iterations), ϵ = $(solver.ϵ), verbose = $(solver.verbose)\n+----------------------------------------------------------+")
    end

    # original code should run until V converges to V*, this yet needs to be implemented
    # for example as: while max(@. abs(newV - oldV)...) > solver.ϵ
    # However this probably would not work, as newV and oldV have different number of elements (arrays of alphas)
    alphavecs = nothing
    for i in 1:solver.max_iterations
        Γ, alphavecs = improve(pomdp, B, Γ, solver)
        B, Bs, early_term = expand(pomdp, B, Bs)
        if solver.verbose
            println("Iteration $(i) executed, belief set contains $(length(Bs)) belief vectors.")
        end
        if early_term
            if solver.verbose
                println("Belief space did not expand. \nTerminating early.")
            end
            break
        end
    end

    if solver.verbose
        println("+----------------------------------------------------------+")
    end
    acts = [alphavec.action for alphavec in alphavecs]
    return AlphaVectorPolicy(pomdp, Γ, acts)
end
