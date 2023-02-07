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
pol = solve(PBVISolver(max_time=1.0, verbose=true, max_iter=20), pomdp)
sol = PBVISolver(max_time=1.0, verbose=false, max_iter=10)
@profiler solve(sol, pomdp)

@btime solve(sol, pomdp)

PBVI.belief_value(tree.Γ, tree.b[1])

solver2 = PointBasedValueIteration.PBVISolver(;max_iterations=10)
m2 = RockSamplePOMDP()


pol1 = solve(sol, pomdp)


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

##
import PointBasedValueIteration
sol2 = PointBasedValueIteration.PBVISolver(max_iterations=10, verbose=true)
pol2 = solve(sol2, pomdp)

value(pol2, initialstate(pomdp))
value(pol1, initialstate(pomdp))
