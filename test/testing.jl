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
solver = PBVISolver(;max_iter=2)

##
pol = solve(PBVISolver(max_time=5.0, verbose=true, max_iter=10), pomdp)
sol = PBVISolver(max_time=1.0, verbose=false, max_iter=20)
@profiler solve(sol, pomdp)

@profiler solve(sol, pomdp)


@btime solve(sol, pomdp)

PBVI.belief_value(tree.Γ, tree.b[1])

solver2 = PointBasedValueIteration.PBVISolver(;max_iterations=2)
m2 = RockSamplePOMDP()


pol1 = solve(sol2, m2)

value(pol, initialstate(pomdp))

##
using SparseArrays
using LinearAlgebra
v1 = sprandn(100, 0.5)
v2 = sprandn(100, 0.5)
v3 = similar(v1)

dist1(v1,v2) = norm(v1 .- v2, 2)
dist2(v1, v2) = mapreduce(abs_dist, +, v1, v2)
dist3(v1, v2, v3) = norm(v3 .= v1 .- v2, 2)
abs_dist(a,b) = abs(a-b)

using BenchmarkTools
@btime dist1($v1,$v2)
@btime dist2($v1,$v2)
@btime dist3($v1, $v2, $v3)

mapreduce(-, +, v1, v2)
