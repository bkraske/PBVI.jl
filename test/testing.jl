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



m = TigerPOMDP()
solver = PBVISolver(;max_iter=3)
solve(solver, m)
solver2 = PointBasedValueIteration.PBVISolver(;max_iterations=10)
m2 = RockSamplePOMDP()
