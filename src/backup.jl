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
    return max_α.v
end

function belief_value(Γ, b)
    max_val = -Inf
    for α ∈ Γ
        val = dot(α, b)
        if val > max_val
            max_val = val
        end
    end
    return max_val
end

function backup_a!(α, pomdp::ModifiedSparseTabular, cache::TreeCache, a, Γao)
    γ = discount(pomdp)
    R = @view pomdp.R[:,a]
    T_a = pomdp.T[a]
    Z_a = cache.Oᵀ[a]
    Γa = @view Γao[:,:,a]

    Tnz = nonzeros(T_a)
    Trv = rowvals(T_a)
    Znz = nonzeros(Z_a)
    Zrv = rowvals(Z_a)

    for s ∈ eachindex(α)
        v = 0.0
        for sp_idx ∈ nzrange(T_a, s)
            sp = Trv[sp_idx]
            p = Tnz[sp_idx]
            tmp = 0.0
            for o_idx ∈ nzrange(Z_a, sp)
                o = Zrv[o_idx]
                po = Znz[o_idx]
                tmp += po*Γa[sp, o]
            end
            v += tmp*p
        end
        α[s] = v
    end
    @. α = R + γ*α
end

function backup(tree, b_idx)
    Γ = tree.Γ
    b = tree.b[b_idx]
    pomdp = tree.pomdp
    γ = discount(pomdp)
    S = states(pomdp)
    A = actions(pomdp)
    O = observations(pomdp)

    Γao = tree.cache.Γ

    for a ∈ A
        ba_idx = tree.b_children[b_idx][a]
        for o ∈ O
            bp_idx = tree.ba_children[ba_idx][o]
            bp = tree.b[bp_idx]
            Γao[:,o,a] .= max_alpha_val(Γ, bp)
        end
    end

    V = -Inf
    α_a = tree.cache.alpha # zeros(Float64, length(S))
    best_α = zeros(Float64, length(S))
    best_action = first(A)

    for a ∈ A
        α_a = backup_a!(α_a, pomdp, tree.cache, a, Γao)
        Qba = dot(α_a, b)
        if Qba > V
            V = Qba
            best_α .= α_a
            best_action = a
        end
    end

    return AlphaVec(best_α, best_action)
end

function backup!(tree)
    Γ′ = empty!(tree.cache.Γ_tmp)
    for b_idx ∈ tree.real
        !tree.is_terminal[b_idx] && push!(Γ′,backup(tree, b_idx))
    end
    resize!(tree.Γ, length(Γ′))
    copyto!(tree.Γ, Γ′)
end
