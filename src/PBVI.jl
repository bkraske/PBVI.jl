module PBVI

using POMDPs
using POMDPTools
using LinearAlgebra
using Distributions
using FiniteHorizonPOMDPs

import POMDPs: Solver, solve
import Base: ==, hash, convert
import FiniteHorizonPOMDPs: InStageDistribution, FixedHorizonPOMDPWrapper

export PBVISolver2, solve

include("solver.jl")

end