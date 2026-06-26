import Base: show


"""
    fit(::Type{FlyNNM}, X::AbstractMatrix, y::AbstractVector, P::AbstractProjectionMatrix,
    k::Int, γ::Real) -> FlyNNM

Trains the FlyNNM classifier.

# Arguments
- `::Type{FlyNNM}`: The model to be fitted.
- `X::AbstractMatrix`: Training data matrix (d x n).
- `y::AbstractVector`: Training labels (n-element vector).
- `P::AbstractProjectionMatrix`: Random projection matrix (m x d).
- `k::Int`: The number of nonzeros in the FlyHash.
- `γ::Real`: The decay rate parameter.

# Returns
- `FlyNNM`: The trained model containing the projection matrix, weights,
and class map.
"""
function fit(::Type{FlyNNM}, X::AbstractMatrix, y::AbstractVector, P::AbstractProjectionMatrix, k::Int, γ::Real)
    d_X, n = size(X)
    m, d_P = size(P)

    @assert d_X == d_P "Dimension mismatch: X has $d_X rows, M has $d_P rows."
    @assert length(y) == n "Number of labels does not match number of data points."

    # Robust mapping of class labels to integer indices (1, 2, ..., l).
    # This makes the code work with non-numeric or non-sequential labels.
    class_labels = unique(y)
    l = length(class_labels)
    class_map = Dict(label => i for (i, label) in enumerate(class_labels))

    # Compute the FlyHash
    H = FlyHash(X, P, k).matrix

    tasks = map(chunks(1:n; n=nthreads())) do inds
        @spawn begin
            W_local = zeros(Int32, l, m)

            for i in inds
                class_idx = class_map[@inbounds y[i]]
                @inbounds for j in H.colptr[i]:(H.colptr[i+1]-1)
                    r = H.rowval[j]
                    W_local[class_idx, r] += 1
                end
            end
            return W_local
        end
    end

    # Reduction (combination) of results.
    # Once all threads are finished, we combine their local weight matrices.
    W_counts = sum(fetch.(tasks))

    # Apply the final weight transformation with special handling for γ = 0.
    W_final = if γ == 0
        # If γ is 0, update weights to 1.0 where are 0, and to 0.0 otherwise.
        @. float(W_counts == 0)
    else
        λ = log1p(-γ)  # = log(1-γ)
        @. exp(λ * float(W_counts))
    end

    return FlyNNM(P, W_final, k, class_labels)
end

"""
    predict(model::FlyNNM, X::AbstractMatrix) -> Vector

Performs inference on new data using a trained FlyNNM model.
Ties are broken deterministically by selecting the first class with minimum score.

# Arguments
- `model::FlyNNM`: The trained FlyNNM model object.
- `X::AbstractMatrix`: The data matrix (d x n).

# Returns
- `Vector`: A vector of predicted labels for each column in `X`.
"""
function predict(model::FlyNNM, X::AbstractMatrix)
    H = FlyHash(X, model.P, model.k).matrix
    fX = model.W * H

    l, n = size(fX)
    y_pred = Vector{eltype(model.class_labels)}(undef, n)

    @threads for i in 1:n
        col = @view fX[:, i]

        # Find the index of the minimum value (deterministic tie-break)
        min_val = typemax(eltype(fX))
        winner_idx = 1

        @inbounds for j in 1:l
            val = col[j]
            if val < min_val
                min_val = val
                winner_idx = j
            end
        end

        @inbounds y_pred[i] = model.class_labels[winner_idx]
    end

    return y_pred
end



"""
    show(io::IO, ::MIME"text/plain", model::FlyNNM)

Defines the multi-line, pretty-printing for a FlyNNM model (rich display).
"""
function show(io::IO, ::MIME"text/plain", model::FlyNNM)
    println(io, "FlyNNM Classifier")
    println(io, "├─ Projection (P): $(join(size(model.P), '×')) $(typeof(model.P))")
    println(io, "├─ Weights (W):    $(join(size(model.W), '×')) $(typeof(model.W))")
    println(io, "├─ Nonzeros (k):  $(model.k)")
    println(io, "└─ Classes:        $(length(model.class_labels)) labels of type $(eltype(model.class_labels))")
end

"""
    show(io::IO, model::FlyNNM)

Defines the compact, single-line printing for a FlyNNM model.
"""
function show(io::IO, model::FlyNNM)
    n_classes = length(model.class_labels)
    print(io, "FlyNNM($(n_classes) classes, k=$(model.k))")
end
