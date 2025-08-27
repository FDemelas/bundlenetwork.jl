"""
Innner Loss function for the training of a neural network for image classification of the MNIST dataset.
It works with batches and so `x` and `y` are, respectively, the input and the output of the batch.
An input `x` is an 28×28 image, here encoded as a vector and y is the number (between 0 and 9) represented in the image, here encoded as one-hot-vector.
The loss function considered in the 'inner' learning task is here denoted by `loss`, that is rescaled dividing it by `rescaling_factor`.
The neural network has `length(layers)` hidden layers. The `i`-th layer has size `layers[i]`. 
Note: This function is not really concave, but it could be used to test our package for Meta-Learning.
"""
struct InnerLoss <: AbstractConcaveFunction
	x::Any
	y::Any
	loss::Any
	layers::Any
	rescaling_factor::Any
end

"""
Forward-pass for the `InnerLoss`.
It computes the output of the network and compute the loss function between the prediction and the labels.
As said in the definition of `InnerLoss` inputs and labels corresponds to a batch.
Note: this function takes as input the parameters of the 'hidden' network and compute the associated prediction and then provides the loss value of the former.
"""
function (l::InnerLoss)(params)
	h, y = l.x, l.y
	first = 1
	for i in 1:(length(l.layers)-1)
		W = copy(reshape(params[first:(first+prod(l.layers[i:i+1])-1)], l.layers[i:i+1]...))
		first += prod(l.layers[i:i+1]) - 1
		b = copy(params[first:first+l.layers[i+1]-1])
		h = copy(sigmoid.(W' * h .+ b))
		first += l.layers[i+1]
	end
	ŷ = h
	return l.loss(ŷ, y)/l.rescaling_factor
end

Flux.@layer InnerLoss

"""
This function should be used only during inference for the `InnerLoss` in order to have only the output predicted by the network but not compute the loss function.
Note: this function takes as input the `InnerLoss` structure and the parameters of the 'hidden' network and compute the associated prediction.
"""
function prediction(l,params)	
	h, y = l.x, l.y
	first = 1
	for i in 1:(length(l.layers)-1)
		W = copy(reshape(params[first:(first+prod(l.layers[i:i+1])-1)], l.layers[i:i+1]...))
		first += prod(l.layers[i:i+1]) - 1
		b = copy(params[first:first+l.layers[i+1]-1])
		h = copy(sigmoid.(W' * h .+ b))
		first += l.layers[i+1]
	end
	return softmax(h)
end


"""
Constructor for an `InnerLoss`.
It takes as input the input/output data (`data`) for the MNIST dataset, a rescaling factor to divide the loss value (`rescaling_factor`) and a vector that contains the sizes of the layers (`layers`) .
"""
function constructFunction(data, rescaling_factor::Real, layers = [28 * 20, 20, 10])
	x, y = data
	y = hcat([[j == y[i] for j in 0:9] for i in eachindex(y)]...)
	x = reshape(x, prod(size(x)[1:2]), size(x)[3])
    f(i,j)=-Flux.logitcrossentropy(i,j)
	return InnerLoss(device(x), device(y), f, cpu(layers), rescaling_factor)
end

"""
Size of the input of (the `InnerLoss`) `l`.
In practice it corresponds to the size of the parameters of the 'hidden' network.
"""
function sizeInputSpace(l::InnerLoss)
	return sum([prod(l.layers[i:i+1]) + l.layers[i+1] for i in 1:length(l.layers)-1])
end

function numberSP(l::InnerLoss)
	return 1
end


"""
Given an `AbstractConcaveFunction` `ϕ` and an input vector `z` for the former, this function computes the value and a sub-gradient for `ϕ(z)`.
"""
function value_gradient(ϕ::InnerLoss,z::AbstractArray)
    obj,grad=Flux.withgradient((x)->ϕ(x),device(z))
    return device(obj), device(grad[1])
end
