using POMDPs
using POMDPTools
using POMDPModels
import PointBasedValueIteration
using BenchmarkTools
using RockSample

m = TigerPOMDP()
solver = PBVISolver(;max_iterations=10)
solver2 = PointBasedValueIteration.PBVISolver(;max_iterations=10)
m2 = RockSamplePOMDP()