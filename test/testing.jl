using POMDPs
using POMDPTools
using POMDPModels
import PointBasedValueIteration
using BenchmarkTools

m = TigerPOMDP()
solver = PBVISolver2()
solver2 = PointBasedValueIteration.PBVISolver()