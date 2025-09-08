"""
Create the Dual Master Problem associated to the Bundle `B` using `t` as regularization parameter.
It is suggested to use this function only in the initialization and then consider to update the model instead of creating a new one.
"""
function create_DQP(B::DualBundle, t::Float64)
	# Create the (empty) model and assign Gurobi as Optimizer
	model = Model(Gurobi.Optimizer)
	set_optimizer_attribute(model, "NonConvex", 0)
	set_optimizer_attribute(model, "FeasibilityTol", 0.01)
	set_optimizer_attribute(model, "MIPGap", 0.01)
	#set_time_limit_sec(model, 1.0)
	# Use only one Thread for the resolution
	#set_attribute(model, "Threads", 1)
	# Force to use the Primal Method (as it allows better re-optimization)
	#set_attribute(model, "Method", 0)

	# Define the objective values for the quadratic part
	g = 0 < B.params.max_β_size < Inf ? B.G[:, 1:B.size] : B.G

	# Define the objective values for the linear part
	α = 0 < B.params.max_β_size < Inf ? B.α[1:B.size] : B.α

	# Define one variable for each bundle component
	# Bound it in 0-1 as it should belong to the simplex
	@variable(model, 1 >= θ[1:size_Bundle(B)[1]] >= 0)

	# add Simplex constraint
	@constraint(model, conv_comb, ones(size_Bundle(B)[1])' * θ == 1)

	# Define the objective function as sum of the linear and the quadratic part
	# Note that the regularization parameter `t` is considered as `1/t` in the linear part (instead of `t` in the quadratic part)
	# as this allows faster re-optimization
	if B.sign
		@variable(model, λ[1:size(B.z[:, 1], 1)] >= 0)
		rhs_value = -zS(B)
		@constraint(model, non_negativity[i = 1:size(B.G, 1)], t * (g[i] * θ[1] + λ[i]) >= rhs_value[i])
	end
	quadratic_part = B.sign ? @expression(model, LinearAlgebra.dot(g * θ + λ, g * θ + λ)) : @expression(model, LinearAlgebra.dot(g * θ , g * θ ))
	linear_part = B.sign ? @expression(model, LinearAlgebra.dot(α, θ) + LinearAlgebra.dot(λ, zS(B))) : @expression(model, LinearAlgebra.dot(α, θ))

	@objective(model, Min, (1 / 2) * quadratic_part .+ 1 / t * linear_part)
	model.obj_dict = B.sign ? Dict(
		:θ => θ,                         # your simplex vars
		:λ => λ,
	) : Dict(
		:θ => θ,                         # your simplex vars
	)
	# the model should not provide output in the standarrd output
	set_silent(model)
	return model
end


"""
Solve the Dual Master Problem and update the associated objective function.
Note: the objective function is rescaled using the regularization parameter as it is scaled inversely in the Dual Master Problem formulation in order to allow faster re-optimization.
"""
function solve_DQP(B::DualBundle)
	#Solve the Dual Quadratic (Master) Problem
	optimize!(B.model)

	#Compute the objective value and scale it back as the regularization parameter was considered as `1/t` in the linear part. 
	B.objB = B.params.t * JuMP.objective_value(B.model)
end

"""
Update the Dual Master Problem formulation.
We check if we add a new component or not.
The second case happens when we found a point for which the associated sub-gradient was equal to the one of an already visited point.
In this case we did not "add" a new point, but just update the information and so the reoptimization could be easier.
If we add a new component, then we add the associated varibale to the problem, by creating a new variable adding it to the simplex constraints and to the objective function.
If we changed the parameter t or the stabilization point we have to update all the linear part in the objective function.
"""
function update_DQP!(B::DualBundle, t_change = true, s_change = true)
	#Smart updates of the Dual Master Problem to improve reoptimization
	if !(length(B.model.obj_dict[:θ]) == B.size)
		# if we add new components to the bundle, then we have to add the associated variable to the Dual Master Problem
		# and add them to the simplex constraint
		for _ in 1:(size_Bundle(B)-length(B.model.obj_dict[:θ]))
			# add a new variable to the bundle
			θ_tmp = @variable(B.model, upper_bound = 1, lower_bound = 0)
			# add the variable to the constrain assuring that θ is in the simplex
			set_normalized_coefficient(JuMP.constraint_by_name(B.model, "conv_comb"), θ_tmp, 1)
			# Add the correspondent terms to the objective function
			# Update the quadratic part
			for (idx, θ) in enumerate(B.model.obj_dict[:θ])
				set_objective_coefficient(B.model, θ, θ_tmp, 2 / 2 * B.Q[idx, B.li])
			end
			# add the dependence of the variable with respect to itself
			set_objective_coefficient(B.model, θ_tmp, θ_tmp, 1 / 2 * B.Q[B.li, B.li])
			# Update the linear part
			set_objective_coefficient(B.model, θ_tmp, 1 / (B.params.t) * B.α[B.li])
			if B.sign
				# if we are considering non-negative multipliers 
				# we have to add the variable to each non-negativity constraint
				for i in 1:size(B.G, 1)
					set_normalized_coefficient(
						JuMP.constraint_by_name(B.model, "non_negativity[" * string(i) * "]"),
						θ_tmp,
						(B.params.t) * B.G[i, B.li],
					)
				end
				# and update the objective function accordingly
				for (idx, λ) in enumerate(B.model.obj_dict[:λ])
					set_objective_coefficient(B.model, λ, θ_tmp, B.G[idx, B.li])
				end
			end
			# add the variable to the dictionary
			push!(B.model.obj_dict[:θ], θ_tmp)
		end
	end
	# if the stabilization point has changed
	if s_change
		# Update the linear part of the objective function
		set_objective_coefficient(B.model, B.model.obj_dict[:θ], 1 / B.params.t * B.α)
		# change the right hand side of the non negativity constraints
		if B.sign
			rhs_value = -Float64.(zS(B))
			for i in 1:size(B.G, 1)
				set_normalized_rhs(JuMP.constraint_by_name(B.model, "non_negativity[" * string(i) * "]"), rhs_value[i])
			end
		end
	end
	# if the t parameter or the stabilization point has changed
	if t_change
		# Update the linear part of the objective function
		set_objective_coefficient(B.model, B.model.obj_dict[:θ], 1 / B.params.t * B.α)
		if B.sign
			m = size(B.G, 1)
			for i in 1:m
				set_normalized_coefficient(
					JuMP.constraint_by_name(B.model, "non_negativity[" * string(i) * "]"),
					B.model.obj_dict[:λ][i],
					(B.params.t),
				)
				for j in eachindex(B.model.obj_dict[:θ])
					set_normalized_coefficient(
						JuMP.constraint_by_name(B.model, "non_negativity[" * string(i) * "]"),
						B.model.obj_dict[:θ][j],
						(B.params.t) * B.G[i, j],
					)
				end
			end
		end
	end
end

"""
Returns the linearization error associated to the stabilization point.
"""
function αS(B::DualBundle)
	return B.α[B.s]
end

"""
Returns the sub-gradient associated to the stabilization point.
"""
function gS(B::DualBundle)
	return B.G[:, B.s]
end

"""
Returns the stabilization point.
"""
function zS(B::DualBundle)
	return B.z[:, B.s]
end

"""
Returns the objective function associated to the stabilization point.
"""
function objS(B::DualBundle)
	return B.obj[B.s]
end

"""
Returns the size of the bundle.
"""
function size_Bundle(B::DualBundle)
	return B.size
end

"""
Returns the linearization error associated to the point in the i-th position in the bundle.
"""
function linearization_error(B::DualBundle, i::Int)
	if i == B.s
		# avoid computations: the linearization error for the stabilization point is zero
		return 0
	end
	return linearization_error(B.G[:, i], zS(B), B.z[:, i], objS(B), B.obj[i])
end

"""
Returns the linearization error associated to the point `z` (that has the gradient `g` and objective value `obj`) with respect to the stabilization point `zS` (that has objective value `objS`).
"""
function linearization_error(g::AbstractVector, zS::AbstractVector, z::AbstractVector, objS::Real, obj::Real)
	return g' * (zS - z) - (objS - obj)
end

"""
Updates all the linearization errors in the Bundle.
"""
function update_linearization_errors(B::DualBundle)
	for i in 1:size_Bundle(B)
		B.α[i] = linearization_error(B, i)
	end
end

"""
Removes components from the Bundle that are not used for many consecutives iterations.
"""
function remove_outdated(B::DualBundle, ϵ = 1e-6)
	sB = size_Bundle(B)
	remove_idx = []
	keep_idx = []
	for i in 1:sB
		if i == B.s
			append!(keep_idx, i)
		else
			keep = 0
			how_much_iter_in = 0
			for j in eachindex(B.cumulative_θ)
				if i <= length(B.cumulative_θ[j])
					keep += (B.cumulative_θ[j][i] > ϵ)
					how_much_iter_in += 1
				end
			end
			if keep > 0 || (how_much_iter_in <= B.params.remotionStep)
				append!(keep_idx, i)
			else
				append!(remove_idx, i)
				B.size -= 1
			end
		end
	end
	if 0 < B.params.max_β_size < Inf
		B.size = length(keep_idx)
		while (B.size > B.params.max_β_size)
			θ = B.θ[keep_idx]
			i = argmin(θ)
			append!(remove_idx, keep_idx[i])
			popat!(keep_idx, keep_idx[i])
			B.size -= 1
		end
	end
	first_idxs = collect(1:length(keep_idx))
	if 0 < B.params.max_β_size < Inf
		first_idxs = collect(1:length(keep_idx))
		B.G[:, first_idxs] = B.G[:, keep_idx]
		B.Q[first_idxs, first_idxs] = B.Q[keep_idx, keep_idx]
		B.α[first_idxs] = B.α[keep_idx]
		B.z[:, first_idxs] = B.z[:, keep_idx]
		B.obj[first_idxs] = B.obj[keep_idx]
		B.θ = B.θ[keep_idx]
		B.size = length(keep_idx)
	else
		B.G = B.G[:, keep_idx]
		B.Q = B.Q[keep_idx, keep_idx]
		B.α = B.α[keep_idx]
		B.z = B.z[:, keep_idx]
		B.obj = B.obj[keep_idx]
		B.size = length(keep_idx)
		B.θ = B.θ[keep_idx]
	end
	sort!(remove_idx, rev = true)
	for h in remove_idx
		if h < B.li
			B.li -= 1
		end
		if h < B.s
			B.s -= 1
		elseif h == B.s
			println("Trying to remove stabilization point")
		end
		delete(B.model, B.model.obj_dict[:θ][h])
		deleteat!(B.model.obj_dict[:θ], h)
		for i in 1:size(B.cumulative_θ, 1)
			if h < size(B.cumulative_θ[i], 1)
				deleteat!(B.cumulative_θ[i], h)
			end
		end
	end
	B.cumulative_θ = B.cumulative_θ[max(1, end - B.params.remotionStep):end]

end

"""
Updates the solution `B.θ` of the Dual Master Problem. 
Compute the new searching direction `B.w`, obtained as convex combination of the gradients contained in the bundle with weights `B.θ`.
It also stock sever quantities that will be used in the condition that determinines if change the stailization point (i.e. make a Serious Step or a Null Step) and in the t-strategies.
"""
function compute_direction(B::DualBundle)
	# Obtain the solution of the Dual Master Problem. It is an element of the simplex for which each component denotes the weight for the associated bundle gradient
	B.θ = value.(B.model.obj_dict[:θ])
	# Add this solution into memory to allow removing outdated components
	push!(B.cumulative_θ, copy(B.θ))
	# Compute the new trial direction as convex combination of the gradients in the bundle
	B.w = (0 < B.params.max_β_size < Inf) ? (B.G[:, 1:B.size] * B.θ) : (B.G * B.θ)
	# Compute the value of the linear part (to be used in SS/NS decision and t-strategies)
	B.linear_part = B.α[1:B.size]'B.θ
	# Compute the value of the quadratic part (to be used in SS/NS decision and t-strategies)
	B.quadratic_part = B.w'B.w
	# Compute the Dual Master Problem objective value (to be used in SS/NS decision and t-strategies)
	B.vStar = (B.params.t * B.quadratic_part + B.linear_part)
	# Compute the Dual Master Problem objective value with a different (fixed) t, but keeping the sam solution (to be used in SS/NS decision and t-strategies)
	B.ϵ = B.linear_part + B.params.t_star * B.quadratic_part / 2
end

"""
Updates the Bundle information.
It adds to the bundle the information associated to the stabilization point `z`, knowing that the objective value in this point is `obj` and the sub-gradient is `g`.
"""
function update_Bundle(B::DualBundle, z, g, obj)
	# reshape the new trial point as a vector
	z = reshape(z, :)
	# and also the associated gradient
	g = Float32.(reshape(g, :))

	already_exists = false
	for j in 1:B.size
		#check if the new gradient already exists in the gradient matrix
		if (sum(abs.(B.G[:, j] - g)) < 1.0e-3)
			already_exists = true
			# if it already exists we does not need to add it, 
			# we can just update the associated linearization error, the objective value and the new point
			B.α[j] = linearization_error(g, zS(B), z, objS(B), obj)
			B.obj[j] = obj
			B.z[:, j] = z

			# update the parameter to denote the last inserted components to denote that component
			# Note: it is not mandatory the last one
			B.li = j

			#Add the objective value to the vector memorizing all the objective values
			push!(B.all_objs, obj)
			# in this case the update finish here
			return
		end

	end

	# if the component is actually a "new one", in the sense that the associated sub-gradient is "new"
	if !(already_exists)
		# two different implementations are proposed the first one use values of fixed-size for the fields of b
		# the second one of variable sizes
		if 0 < B.params.max_β_size < Inf
			i = B.size + 1
			# add the new objective, the new point, the new gradent and the linearization error
			B.obj[i] = obj
			B.z[:, i] = z
			B.G[:, i] = g
			B.α[i] = linearization_error(B, i)

			# upated the matrix Q=G'G
			q = B.G[:, 1:B.size]' * g
			B.Q[1:i-1, i] = q
			B.Q[1, 1:i-1] = q
			B.Q[i, i] = g'g
		else
			# upated the matrix Q=G'G
			# and add the new objective, the new point, the new gradent and the linearization error
			B.z = hcat(B.z, z)
			q = B.G' * g
			B.G = hcat(B.G, g)
			B.Q = vcat(hcat(B.Q, q), vcat(q, g'g)')
			push!(B.obj, obj)
			α = linearization_error(B, size_Bundle(B) + 1)
			push!(B.α, α)
		end
		# set the correct index for the last inserted index
		B.li = B.size + 1
		# add the objective value to the vector that memorize the objective values at all the iterations
		push!(B.all_objs, obj)
		# increment the bundle size
		B.size += 1
	end
end

"""
Return `true` is the stopping criteria is satisfied and `false` if not.
The stopping criteria require that the sum of the quadratic part (weighted with the hyper-parameter `t_star`) and the linear part are small.
"""
function stopping_criteria(B::DualBundle)
	return B.params.t_star * B.quadratic_part + B.linear_part <= B.params.ϵ * (max(0, objS(B)) + 1)
end

"""
Computes the new trial point by moving from the stabilization point `zS(B)` to a step `B.params.t` through the direction `B.w`.
"""
function trial_point(B::DualBundle)
	tp = zS(B) + B.params.t * (B.w .+ (B.sign ? reshape(value.(B.model.obj_dict[:λ]), :) : 0.0))
	return B.sign ? relu(tp) : tp
end

"""
Main function for the Bundle.
It maximize the function `ϕ` using the bundle method, with a previously initialized bundle `B`.
It has two additional input parameters:
- `t_strat`: the t-Strategy, by default it is the contant t-strategy (i.e. the regularization parameter for the Dual Master Problem is always keeped fixed).
- `unstable`: if `true` we always change the stabilization point using the last visited point, by default `false` (change it only if you know what you are doing).
"""
function solve!(B::DualBundle, ϕ::AbstractConcaveFunction; t_strat::abstract_t_strategy = constant_t_strategy(), unstable::Bool = false, force_maxIt::Bool = true)
	times = Dict("times" => [], "trial point" => [], "ϕ" => [], "update β" => [], "SS/NS" => [], "update DQP" => [], "solve DQP" => [], "remove outdated" => [])
	t0 = time()
	for epoch in 1:B.params.maxIt
		t1 = time()

		# compute the new trial point
		z = trial_point(B)
		append!(times["trial point"], (time() - t1))

		t0 = time()
		# compute the objective value and sub-gradient in the new trial-point
		obj, g = value_gradient(ϕ, z) # to optimize
		g = g
		append!(times["ϕ"], (time() - t0))

		t0 = time()
		# update the bundle with the new information
		update_Bundle(B, z, g, obj)

		# memorize the new regularization parameter
		t = B.params.t
		# memorize the current stabilization point
		s = B.s
		# update the regularization parameter and the stabilization point
		t_strategy(B, B.li, t_strat, unstable)
		append!(times["update β"], (time() - t0))

		t0 = time()
		#update the Dual Master Problem
		update_DQP!(B, t == B.params.t, s == B.s)
		append!(times["update DQP"], (time() - t0))

		t0 = time()
		#solve the Dual Master Problem
		solve_DQP(B) # to optimize
		append!(times["solve DQP"], (time() - t0))

		# Update the Dual Master problem solution and compute the new trial direction
		compute_direction(B)

		# remove outdated components, i.e. bundle components that are not used for many iterations
		t0 = time()
		if epoch >= B.params.remotionStep
			#remove_outdated(B)
		end
		append!(times["remove outdated"], (time() - t0))

		append!(B.ts, B.params.t)
		# check the stopping criteria and if it is satisfied stop the execution
		# if force_maxIt is true than we force the algorithm to attain the maximum iterations
		# and no further stopping criteria is considered 
		if !(force_maxIt) && stopping_criteria(B)
			println("Satisfied stopping criteria")
			return true, times
		end
		push!(B.memorized["times"], time() - t1)
	end
	times["times"] = B.memorized["times"]
	return false, times
end

"""
Function that handle is keep or change the stabilization point and then handle the increases or decreases of the regularization parameter.
"""
function t_strategy(B::DualBundle, i::Int, ts::abstract_t_strategy, unstable::Bool = false)
	# if the new objective improve `enough` the one in the stabilization point
	if B.obj[i] - objS(B) >= B.params.m1 * B.vStar || unstable
		# update the consecutive  SS and NS counters
		B.CSS += 1
		B.CNS = 0
		# change the stabilization point
		B.s = i
		# update the linearization error consequently
		update_linearization_errors(B)
		# call the t-strategy that decide if change the parameter t
		increment_t(B, ts)
	else
		# update the consecutive  SS and NS counters
		B.CNS += 1
		B.CSS = 0
		# call the t-strategy that decide if change the parameter t
		decrement_t(B, ts)
	end
end