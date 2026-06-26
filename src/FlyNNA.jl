import Base: show


"""
    fit(::Type{FlyNNA}, X::AbstractMatrix, y::AbstractVector, P::AbstractProjectionMatrix,
        k::Int) -> FlyNNA

Trains the FlyNNA classifier.

# Arguments
- `::Type{FlyNNA}`: The model to be fitted.
- `X::AbstractMatrix`: Training data matrix (d x n).
- `y::AbstractVector`: Training labels (n-element vector).
- `P::AbstractProjectionMatrix`: Random projection matrix (m x d).
- `k::Int`: The number of active response regions per item.

# Returns
- `FlyNNA`: The trained model containing the projection matrix and weights.
"""
function fit(::Type{FlyNNA}, X::AbstractMatrix{T}, y::AbstractVector, P::AbstractProjectionMatrix, k::Int) where T
    d_X, n = size(X)
    m, d_P = size(P)

    @assert d_X == d_P "Dimension mismatch: X has $d_X rows, M has $d_P rows."
    @assert length(y) == n "Number of labels does not match number of data points."

    # Robust mapping of class labels to integer indices (1, 2, ..., l).
    # This makes the code work with non-numeric or non-sequential labels.
    class_labels = unique(y)
    l = length(class_labels)
    class_map = Dict(label => i for (i, label) in enumerate(class_labels))

    H = FlyHash(X, P, k).matrix

    # Each task computes a partial W and count vector
    tasks = map(chunks(1:n; n=nthreads())) do inds
        @spawn begin
            # Task-local accumulators
            W_local = zeros(Int, l, m)
            ct_local = zeros(Int, m)

            for i in inds
                class_idx = class_map[@inbounds y[i]]
                @inbounds for j in H.colptr[i]:(H.colptr[i+1]-1)
                    r = H.rowval[j]
                    W_local[class_idx, r] += 1
                    ct_local[r] += 1
                end
            end
            return (W_local, ct_local)
        end
    end

    results = fetch.(tasks)

    # Reduction step
    W_total = sum(first.(results))
    ct_total = sum(last.(results))

    W_normalized = zeros(Float64, l, m)
    valid_indices = ct_total .> 0

    @inbounds for c in 1:l
        W_normalized[c, valid_indices] .= W_total[c, valid_indices] ./ ct_total[valid_indices]
    end

    return FlyNNA(P, W_normalized, ct_total, k, class_labels)
end

"""
    predict(model::FlyNNA, X::AbstractMatrix) -> Vector

Performs inference on new data using a trained FlyNNA model.
Ties are broken deterministically by selecting the first class with maximum score.

# Arguments
- `model::FlyNNA`: The trained FlyNNA model object.
- `X::AbstractMatrix`: The data matrix (d x n).

# Returns
- `Vector`: A vector of predictions for each column in `X`.
"""
function predict(model::FlyNNA, X::AbstractMatrix{T}) where T
    n = size(X, 2)
    l, m = size(model.W)

    y_pred = Vector{eltype(model.class_labels)}(undef, n)
    T_proj = promote_type(T, eltype(model.P.matrix))

    tasks = map(chunks(1:n; n=nthreads())) do inds
        @spawn begin
            # Allocate buffers once per task
            x_proj = Vector{T_proj}(undef, m)
            top_idxs = Vector{Int}(undef, model.k)
            top_vals = Vector{T_proj}(undef, model.k)
            class_scores = Vector{eltype(model.W)}(undef, l)

            for i in inds
                # Re-use buffers for every item in this chunk
                mul!(x_proj, model.P, view(X, :, i))
                _topk_indices!(top_idxs, top_vals, x_proj, model.k)

                fill!(class_scores, 0)
                # View into the weights for active indices
                active_weights = view(model.W, :, top_idxs)
                sum!(class_scores, active_weights)

                winner_idx = argmax(class_scores)
                @inbounds y_pred[i] = model.class_labels[winner_idx]
            end
        end
    end
    foreach(wait, tasks)

    return y_pred
end


"""
    show(io::IO, ::MIME"text/plain", model::FlyNNA)

Defines the multi-line, pretty-printing for an FlyNNA model (rich display).
"""
function show(io::IO, ::MIME"text/plain", model::FlyNNA)
    println(io, "FlyNNA Classifier")
    println(io, "├─ Projection (P): $(join(size(model.P), '×')) $(typeof(model.P))")
    println(io, "├─ Weights (W):    $(join(size(model.W), '×')) $(typeof(model.W))")
    println(io, "├─ Activations (k): $(model.k)")
    println(io, "└─ Classes:        $(length(model.class_labels)) labels of type $(eltype(model.class_labels))")
end


"""
    show(io::IO, model::FlyNNA)

Defines the compact, single-line printing for an FlyNNA model.
"""
function show(io::IO, model::FlyNNA)
    n_classes = length(model.class_labels)
    print(io, "FlyNNA($(n_classes) classes, k=$(model.k))")
end
