# Begin module
module FlyClassifiers

# Dependencies
using LinearAlgebra
using SparseArrays
using StatsBase
using Random
using Base.Threads: @spawn, nthreads, @threads
using ChunkSplitters: chunks

abstract type AbstractProjectionMatrix{T} <: AbstractMatrix{T} end
include("projections/RandomBinaryProjectionMatrix.jl")
include("projections/RandomUniformProjectionMatrix.jl")

# Files inclusion
include("types.jl")
#include("utils.jl")
include("FlyHash.jl")
include("FlyNNM.jl")
include("FlyNNA.jl")

# Export of public types and functions
export AbstractFliesClassifier, FlyNNM, FlyNNA, FlyHash
export AbstractProjectionMatrix, RandomBinaryProjectionMatrix, RandomUniformProjectionMatrix
export fit, predict

end # module
