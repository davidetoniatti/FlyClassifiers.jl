import LinearAlgebra: mul!

"""
    RandomBinaryProjectionMatrix

A type representing a random binary projection matrix.
"""
struct RandomBinaryProjectionMatrix <: AbstractProjectionMatrix{Bool}
    matrix::SparseMatrixCSC{Bool,Int}
end

"""
    RandomBinaryProjectionMatrix(d::Int, m::Int, s::Int, seed::Int) ->
    RandomBinaryProjectionMatrix

Constructor for a binary projection matrix of size `m x d`, where each row has
exactly `s` non-zero elements, placed in random columns.

# Arguments
- `d::Int`: Number of rows.
- `m::Int`: Number of columns.
- `s::Int`: Number of non-zero elements per columns.
- `seed::Int`: Seed for initializing the random number generators.
"""
function RandomBinaryProjectionMatrix(m::Int, d::Int, s::Int; seed::Int=42)

    nnz = m * s
    row_idx = Vector{Int}(undef, nnz)
    col_idx = Vector{Int}(undef, nnz)

    # Use multithreading to generate rows in parallel.
    tasks = map(chunks(1:m; n=nthreads())) do inds
	@spawn begin
	    chunk_seed = seed + first(inds)
	    rng = Xoshiro(chunk_seed)
	    # Task-local buffer for sampling indices
	    idxs = Vector{Int}(undef, s)

	    for i in inds
		start_pos = (i - 1) * s + 1
		end_pos = i * s

		# Sampling.
		sample!(rng, 1:d, idxs; replace=false)

		# Fill the row and column index vectors.
		@inbounds row_idx[start_pos:end_pos] .= i
		@inbounds col_idx[start_pos:end_pos] .= idxs
	    end
	end
    end

    foreach(wait, tasks)

    # The values are all `true`, so we can construct the sparse matrix directly.
    sp_mat = sparse(row_idx, col_idx, true, m, d)
    return RandomBinaryProjectionMatrix(sp_mat)
end

# Implement the AbstractArray interface
Base.size(p::RandomBinaryProjectionMatrix) = size(p.matrix)
Base.getindex(p::RandomBinaryProjectionMatrix, i::Int, j::Int) = getindex(p.matrix, i, j)


"""
    mul!(y, P, x) -> y

Performs an efficient, in-place matrix-vector multiplication `y = P * x`.

This function is highly optimized for cases where `P` is a `SparseMatrixCSC{Bool, Int}`.
It leverages the Compressed Sparse Column (CSC) format for fast iteration. For each
non-zero element at `P[i, j]`, the corresponding value `x[j]` is added to the
result vector at `y[i]`. This avoids unnecessary multiplications by zero.

# Arguments
- `y::Vector{T}`: The output vector where the result is stored.
- `P::RandomMatrixCSC{Bool, Int}`: The sparse projection matrix (`m x d`).
- `x::AbstractVector{T}`: The input vector of length `d`.
"""
function mul!(y::Vector{T}, P::RandomBinaryProjectionMatrix, x::AbstractVector{T}) where T
    m, d = size(P)

    @assert length(y) == m "Output vector length must match the number of rows in M."
    @assert length(x) == d "Input vector length must match the number of columns in M."

    # Initialize the output vector to zero to ensure a clean slate for accumulation.
    fill!(y, zero(T))

    colptr = P.matrix.colptr
    rowval = P.matrix.rowval

    # Iterate through each column 'j' of the matrix M.
    # This is the most efficient way to traverse a CSC matrix.
    @inbounds for j in 1:d
        # Get the value from the input vector corresponding to the current column.
        val_x = x[j]

        # Iterate through the non-zero elements of column 'j'.
        # The range `colptr[j]:(colptr[j+1] - 1)` gives the indices in `rowval`
        # for all non-zero entries in this column.
        for k in colptr[j]:(colptr[j+1]-1)
            # Get the row index 'i' of the current non-zero element.
            i = rowval[k]

            # Accumulate the result: y[i] += M[i, j] * x[j].
            # Since M is a boolean matrix, M[i, j] is effectively 1,
            # so we just add x[j] to y[i].
            y[i] += val_x
        end
    end
    return y
end
