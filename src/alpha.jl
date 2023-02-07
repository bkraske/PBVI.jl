struct AlphaVec <: AbstractVector{Float64}
    v::Vector{Float64}
    a::Int
end

@inline Base.length(v::AlphaVec) = length(v.v)

@inline Base.size(v::AlphaVec) = size(v.v)

@inline Base.getindex(v::AlphaVec, i) = getindex(v.v,i)
