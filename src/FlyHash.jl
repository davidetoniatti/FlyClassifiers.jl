"""
    FlyHash(X::AbstractMatrix, P::AbstractProjectionMatrix, k::Int) ->
    FlyHash

Computes the FlyHash of each column of matrix `X`, using the projection matrix `P`.
For each column, the FlyHash is defined by the indices of the `k` largest
projections.

# Arguments
- `X::AbstractMatrix`: Input matrix (d x n).
- `P::AbstractProjectionMatrix`: Random projection matrix (m x d).
- `k::Int`: Number of nonzeros per column (the "top-k").

# Returns
- `SparseMatrixCSC{Bool, Int}`: The sparse FlyHash matrix (m x n).
"""
function FlyHash(X::AbstractMatrix, P::AbstractProjectionMatrix, k::Int)
    d_X, n = size(X)
    m, d_P = size(P)

    @assert d_X == d_P "Dimension mismatch: X has $d_X rows, M has $d_P rows."

    # Determine the computation type based on the element types of X and M.
    T = promote_type(eltype(X), eltype(P.matrix))


    nnz = n * k
    row_idx = Vector{Int}(undef, nnz)
    col_idx = Vector{Int}(undef, nnz)

    tasks = map(chunks(1:n; n=nthreads())) do inds
	@spawn begin
	    # Buffers are now task-local, not thread-local.
	    # Safe regardless of migration.
	    x_proj = Vector{T}(undef, m)
	    top_idxs = Vector{Int}(undef, k)
	    top_vals = Vector{T}(undef, k)

	     for i in inds
		 start_pos = (i - 1) * k + 1
		 end_pos = i * k

		 # In-place multiplication
		 mul!(x_proj, P, X[:, i])

		 # Find the indices of the k largest values.
		 _topk_indices!(top_idxs, top_vals, x_proj, k)
		 #partialsortperm!(top_idxs, x_proj, k; rev=true)

		 @inbounds row_idx[start_pos:end_pos] .= top_idxs
		 @inbounds col_idx[start_pos:end_pos] .= i
	     end
	end
    end

    # Wait for all tasks to complete
    foreach(wait, tasks)

    return FlyHash(sparse(row_idx, col_idx, true, m, n))
end

"""
    _topk_indices!(top_idxs, top_vals, v, k) -> Vector{Int}

Finds the indices of the `k` largest values in a vector `v` and stores them in `top_idxs`.

This is an optimized, in-place function that avoids allocations by using pre-allocated
buffers. It maintains a buffer of the top `k` elements found so far. When it encounters
a new element larger than the current minimum in the buffer, it replaces that minimum
and then performs a linear scan of the small buffer to find the new minimum. This approach
is efficient for small `k`. In fact, it is much faster than partialsortperm!.

# Arguments
- `top_idxs::Vector{Int}`: A pre-allocated buffer of size `k` to store the resulting indices.
- `top_vals::Vector{T}`: A pre-allocated buffer of size `k` to store the corresponding values.
- `v::AbstractVector{T}`: The input vector to search for top elements.
- `k::Int`: The number of top elements to find.

# Returns
- `Vector{Int}`: The modified `top_idxs` buffer containing the indices of the top `k` values.
"""
@inline function _topk_indices!(top_idxs::Vector{Int}, top_vals::Vector{T}, v::AbstractVector{T}, k::Int) where T
    # Initialize the value buffer with the smallest possible value for type T.
    fill!(top_vals, typemin(T))
    # Initialize the index buffer with a placeholder (0 is a safe choice if indices are 1-based).
    fill!(top_idxs, 0)

    # These variables track the minimum value currently in our top-k buffer and its position.
    # This avoids having to search for the minimum every single iteration of the main loop.
    min_val_in_topk = typemin(T)
    min_pos_in_topk = 1

    # Iterate through each element of the input vector `v`.
    @inbounds for j in eachindex(v)
        val = v[j]

        # Check if the current value is larger than the smallest value in our top-k set.
        if val > min_val_in_topk
            # If it is larger, replace the smallest value in our buffer with this new value.
            top_idxs[min_pos_in_topk] = j
            top_vals[min_pos_in_topk] = val

            # We must find the new minimum for the next iteration's comparison.
            # We start the search by assuming the new minimum is the largest possible value.
            min_val_in_topk = typemax(T)

            # Perform a linear scan over the small `k`-sized buffer to find the new minimum.
            for i in 1:k
                if top_vals[i] < min_val_in_topk
                    min_val_in_topk = top_vals[i]
                    min_pos_in_topk = i
                end
            end
        end
    end

    return top_idxs
end
