"""
    FlyHash

A struct to hold the components of a FlyHash.
"""
struct FlyHash
    matrix::SparseMatrixCSC{Bool,Int}
end


"""
    AbstractFliesClassifier

Supertype for FlyNNM and EaS classifiers.
"""
abstract type AbstractFliesClassifier end

"""
    FlyNNM

A struct to hold the components of a trained FlyNNM model.
"""
struct FlyNNM{T,TP<:AbstractProjectionMatrix} <: AbstractFliesClassifier
    P::TP
    W::Matrix{Float64}
    k::Int
    class_labels::Vector{T}
end

"""
    FlyNNA

A struct to hold the components of a trained FlyNNA model.
"""
struct FlyNNA{T,TP<:AbstractProjectionMatrix} <: AbstractFliesClassifier
    P::TP
    W::Matrix{Float64}
    ct::Vector{Int64}
    k::Int
    class_labels::Vector{T}
end