"""
Sparse-max function.

Presented in:

`From Softmax to Sparsemax: A Sparse Model of Attention and Multi-Label Classification`
André F. T. Martins, Ramón Fernandez Astudillo
"""
function sparsemax(γi; dims=1)
	# sorted input in a decreasing order w.r.t. the dimension `dims`
	z = sort(γi; dims, rev = true)
	# compute the cumulative sum of the elements before (not strictly) the current component w.r.t. the dimension `dims`
	cs = cumsum(z; dims)
	# collect the range w.r.t. the dimension `dims`. In pratice provide the row or columns index of the associated component depending to `dims`
	rng = device(dims == 2 ? repeat(collect(1:size(z, 2))', size(z, 1)) : repeat(collect(1:size(z, 1))', size(z, 2))')
	# By `dims` provide a matrix of boolean with true in the associated component if k⋅zₖ + 1 >  ∑ᵏᵢ zᵢ
	is_gt = (rng .* z .> cs .- 1)
	# the maximum index satisfying the previous property
	k = maximum(rng .* is_gt; dims)
	# compute the rescaling factor τ
	τ = (sum(z .* is_gt ; dims) .- 1) ./ k	
	# with this computations the output can be simply found as
    return relu(γi .- τ)
end

"""
Backward pass for the sparsemax function.
"""
function ChainRulesCore.rrule(::typeof(sparsemax), γi; dims=1)
	# sorted input in a decreasing order w.r.t. the dimension `dims`
	z = sort(γi; dims, rev = true)
	# compute the cumulative sum of the elements before (not strictly) the current component w.r.t. the dimension `dims`
	cs = cumsum(z; dims)
	# collect the range w.r.t. the dimension `dims`. In pratice provide the row or columns index of the associated component depending to `dims`
	rng = device(dims == 2 ? repeat(collect(1:size(z, 2))', size(z, 1)) : repeat(collect(1:size(z, 1))', size(z, 2))')
	# By `dims` provide a matrix of boolean with true in the associated component if k⋅zₖ + 1 >  ∑ᵏᵢ zᵢ
	is_gt = (rng .* z  .> cs .- 1)
	# the maximum index satisfying the previous property
	k = maximum(rng .* is_gt; dims)
	# compute the rescaling factor τ
	τ = (sum(z .* is_gt ; dims) .- 1) ./ k
	# with this computations the output can be simply found as
	val = relu(γi .- τ)
	function loss_pullback(dl)
		non_zeros = device(val .!= 0)
		# analytical computation of the gradient using the previously defined quantities
		return (NoTangent(), dl .- sum(dl * non_zeros'; dims) ./ sum(non_zeros; dims), NoTangent())
	end
	return device(val), device(loss_pullback)
end
