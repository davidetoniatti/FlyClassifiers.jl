using Test
using FlyClassifiers
using Random
using Statistics
using SparseArrays
using LinearAlgebra
using StatsBase: sample!
using CategoricalArrays: categorical

"""
    make_blobs(; d, n_per, ntest_per, seed) -> (X_train, y_train, X_test, y_test)

Builds a small, deterministic, linearly-separable two-class dataset of two
Gaussian blobs. Used as a sanity fixture: any working classifier should reach
near-perfect accuracy on it.

# Arguments
- `d::Int`: Input dimension.
- `n_per::Int`: Number of training points per class.
- `ntest_per::Int`: Number of test points per class.
- `seed::Int`: Seed for the data-generating RNG.

# Returns
- `Tuple`: `(X_train, y_train, X_test, y_test)` with `X` matrices of size `d x n`.
"""
function make_blobs(; d=20, n_per=120, ntest_per=40, seed=123)
    rng = MersenneTwister(seed)
    c1 = randn(rng, d) .+ 2.0
    c2 = randn(rng, d) .- 2.0

    X_train = hcat(c1 .+ 0.5 .* randn(rng, d, n_per), c2 .+ 0.5 .* randn(rng, d, n_per))
    y_train = vcat(fill(1, n_per), fill(2, n_per))
    X_test = hcat(c1 .+ 0.5 .* randn(rng, d, ntest_per), c2 .+ 0.5 .* randn(rng, d, ntest_per))
    y_test = vcat(fill(1, ntest_per), fill(2, ntest_per))

    return X_train, y_train, X_test, y_test
end

"""
    binary_reference(m, d, s; seed) -> SparseMatrixCSC{Bool,Int}

Builds the same matrix as `RandomBinaryProjectionMatrix`, but with a plain
serial loop over the rows. Because each row is seeded independently from
`(seed, row)`, the parallel constructor must return exactly this matrix
regardless of the thread count; comparing against it locks the reproducibility
contract and would catch any regression to per-chunk seeding.
"""
function binary_reference(m::Int, d::Int, s::Int; seed::Int=42)
    row_idx = Vector{Int}(undef, m * s)
    col_idx = Vector{Int}(undef, m * s)
    idxs = Vector{Int}(undef, s)

    for i in 1:m
        rng = Xoshiro(hash((seed, i)))
        sample!(rng, 1:d, idxs; replace=false)
        row_idx[(i-1)*s+1:i*s] .= i
        col_idx[(i-1)*s+1:i*s] .= idxs
    end

    return sparse(row_idx, col_idx, true, m, d)
end

accuracy(y_true, y_pred) = mean(y_true .== y_pred)

const d = 20
const m = 8 * d
const k = 16
const s = 4
const γ = 0.9

@testset "FlyClassifiers.jl" begin

    @testset "Projection matrices: shape and invariants" begin
        B = RandomBinaryProjectionMatrix(m, d, s; seed=42)
        U = RandomUniformProjectionMatrix(m, d; seed=42)

        @test size(B) == (m, d)
        @test size(U) == (m, d)

        # Each row of the binary matrix must have exactly `s` non-zeros.
        @test all(==(s), vec(sum(B.matrix; dims=2)))

        # Each row of the uniform matrix must lie on the unit sphere S^{d-1}.
        @test all(i -> isapprox(norm(@view U.matrix[i, :]), 1.0; atol=1e-8), 1:m)
    end

    @testset "Projection matrices: reproducibility" begin
        # Same seed must give the exact same matrix.
        @test RandomBinaryProjectionMatrix(m, d, s; seed=42).matrix ==
              RandomBinaryProjectionMatrix(m, d, s; seed=42).matrix
        @test RandomUniformProjectionMatrix(m, d; seed=7).matrix ==
              RandomUniformProjectionMatrix(m, d; seed=7).matrix

        # Different seeds must give different matrices.
        @test RandomBinaryProjectionMatrix(m, d, s; seed=1).matrix !=
              RandomBinaryProjectionMatrix(m, d, s; seed=2).matrix

        # The parallel build must match the serial per-row reference, i.e. it is
        # independent of how the rows are chunked across threads.
        @test RandomBinaryProjectionMatrix(m, d, s; seed=42).matrix ==
              binary_reference(m, d, s; seed=42)
    end

    @testset "FlyHash: shape and top-k correctness" begin
        X, _, _, _ = make_blobs()
        n = size(X, 2)
        U = RandomUniformProjectionMatrix(m, d; seed=42)
        H = FlyHash(X, U, k).matrix

        @test size(H) == (m, n)
        # Exactly `k` active units per column.
        @test all(==(k), vec(sum(H; dims=1)))

        # The active units must be the indices of the `k` largest projections.
        proj = Matrix(U.matrix) * X
        for j in 1:n
            expected = Set(partialsortperm(@view(proj[:, j]), 1:k; rev=true))
            actual = Set(H.rowval[H.colptr[j]:(H.colptr[j+1]-1)])
            @test actual == expected
        end
    end

    @testset "fit/predict: all four variants learn" begin
        X_train, y_train, X_test, y_test = make_blobs()
        B = RandomBinaryProjectionMatrix(m, d, s; seed=42)
        U = RandomUniformProjectionMatrix(m, d; seed=42)

        models = Dict(
            "FlyNN-MB" => fit(FlyNNM, X_train, y_train, B, k, γ),
            "FlyNN-MU" => fit(FlyNNM, X_train, y_train, U, k, γ),
            "FlyNN-AB" => fit(FlyNNA, X_train, y_train, B, k),
            "FlyNN-AU" => fit(FlyNNA, X_train, y_train, U, k),
        )

        for (name, model) in models
            y_pred = predict(model, X_test)
            @test length(y_pred) == length(y_test)
            @test eltype(y_pred) == eltype(y_train)
            # Well-separated blobs: every variant should classify them easily.
            @test accuracy(y_test, y_pred) ≥ 0.8
        end
    end

    @testset "Determinism: refit/repredict is identical" begin
        X_train, y_train, X_test, _ = make_blobs()

        for (build_P, do_fit) in (
            (() -> RandomBinaryProjectionMatrix(m, d, s; seed=42),
             P -> fit(FlyNNM, X_train, y_train, P, k, γ)),
            (() -> RandomUniformProjectionMatrix(m, d; seed=42),
             P -> fit(FlyNNA, X_train, y_train, P, k)),
        )
            p1 = predict(do_fit(build_P()), X_test)
            p2 = predict(do_fit(build_P()), X_test)
            @test p1 == p2
        end
    end

    @testset "Non-integer class labels" begin
        X_train, y_int, X_test, _ = make_blobs()
        # Map the integer labels onto symbols to exercise the generic label path.
        y_train = map(v -> v == 1 ? :cat : :dog, y_int)
        U = RandomUniformProjectionMatrix(m, d; seed=42)

        model = fit(FlyNNA, X_train, y_train, U, k)
        y_pred = predict(model, X_test)
        @test eltype(y_pred) == Symbol
        @test all(in((:cat, :dog)), y_pred)
    end

    @testset "CategoricalArray labels (OpenML-style targets)" begin
        # `unique` on a CategoricalArray returns another CategoricalArray, which
        # must still be stored as a plain `Vector` in the model.
        X_train, y_int, X_test, y_test = make_blobs()
        y_train = categorical(map(v -> v == 1 ? "a" : "b", y_int))
        U = RandomUniformProjectionMatrix(m, d; seed=42)

        model = fit(FlyNNA, X_train, y_train, U, k)
        @test model.class_labels isa Vector
        y_pred = predict(model, X_test)
        @test length(y_pred) == length(y_test)
        @test all(p -> String(p) in ("a", "b"), y_pred)
    end

    @testset "Argument validation" begin
        X_train, y_train, _, _ = make_blobs()
        U = RandomUniformProjectionMatrix(m, d; seed=42)

        # Input dimension must match the projection's column count.
        X_bad = randn(d + 1, size(X_train, 2))
        @test_throws AssertionError fit(FlyNNM, X_bad, y_train, U, k, γ)

        # Number of labels must match the number of data points.
        @test_throws AssertionError fit(FlyNNM, X_train, y_train[1:end-1], U, k, γ)

        # Decay rate γ must lie in [0, 1); γ ≥ 1 would produce NaN weights.
        @test_throws AssertionError fit(FlyNNM, X_train, y_train, U, k, 1.0)
        @test_throws AssertionError fit(FlyNNM, X_train, y_train, U, k, -0.5)
    end

end
