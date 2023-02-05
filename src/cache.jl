struct PBVI_cache{S,A}
    spomdp::SparseTabularPOMDP
    ordered_states::Vector{S}
    ordered_actions::Vector{A}
    terminals::Union{Vector{Int64},Vector{Union{}}}
    not_terminals::Vector{Int64}
    b::Vector{Float64}
    Γa::Vector{Vector{Float64}}
    Γao::Vector{Vector{Float64}}
end

function PBVI_cache(pomdp::POMDP)
    spomdp = SparseTabularPOMDP(pomdp)
    ordered_s = ordered_states(pomdp)
    ordered_a = ordered_actions(pomdp)
    terminals = [stateindex(pomdp, s) for s in ordered_s if isterminal(pomdp, s)]
    not_terminals = [stateindex(pomdp, s) for s in ordered_s if !isterminal(pomdp, s)]
    b = Vector{Float64}(undef,length(states(pomdp)))
    Γa = Vector{Vector{Float64}}(undef, length(ordered_a))
    Γao = Vector{Vector{Float64}}(undef, length(ordered_observations(pomdp)))
    return PBVI_cache(spomdp,ordered_s,ordered_a,
                      terminals,not_terminals,b,Γa,Γao)
end
