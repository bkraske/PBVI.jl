using Test
# using PBVI
using SparseArrays
using POMDPModels
using POMDPs

@testset "tiger" begin
    pomdp = TigerPOMDP()
    sol = PBVISolver(max_iter=10, verbose=true)
    alpha_policy = solve(sol, pomdp)
    v0 = value(alpha_policy, initialstate(pomdp))
    @test isapprox(v0, 19.2; atol=0.1)
end

include("sparse_stuff.jl")
