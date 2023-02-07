begin
    using Pkg
    Pkg.activate(joinpath(@__DIR__, ".."))
    using PBVI
    Pkg.activate(@__DIR__)
    using POMDPs
    using POMDPTools
    using POMDPModels
    using BenchmarkTools
    using RockSample
end



pomdp = TigerPOMDP()
solver = PBVISolver(;max_iter=15)

##
pol = solve(PBVISolver(max_time=5.0, verbose=true, max_iter=20), pomdp)

PBVI.belief_value(tree.Γ, tree.b[1])


solver2 = PointBasedValueIteration.PBVISolver(;max_iterations=10)
m2 = RockSamplePOMDP()


solve(solver, pomdp)
