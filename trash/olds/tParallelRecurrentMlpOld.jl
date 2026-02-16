"""
Structure for a `RnnTParallelModel`
"""
mutable struct RnnTParallelModel <: AbstractTModel
	model::Any
	rng::Any
	sample::Bool
	ϵs::AbstractVector
	deviation::AbstractDeviation
	train_mode::Bool
	RnnTParallelModel(model, rng, sample = false, es = [], deviation = NothingDeviation(),train_mode=true) = new(model, rng, sample, es, deviation,train_mode)
end


struct RnnTParallelModelfactory <: AbstractTModelFactory end

function size_output(_::RnnTParallelModelfactory)
	return 1
end

function create_NN(lt::RnnTParallelModelfactory,recurrent_layer=GRUv3, h_decoder::Vector{Int} = [512, 128], h_act = tanh, h_representation::Int = 32, seed::Int = 1, norm::Bool = true,sample=false)
	f_norm(x) = Flux.normalise(x)
	rng = MersenneTwister(seed)
	init = Flux.truncated_normal(Flux.MersenneTwister(seed); mean = 0.0, std = 0.0001)
	encoder_layer = recurrent_layer(size_features(lt) => 2*h_representation)

	i_decoder_layer = Dense(2*h_representation => h_decoder[1], h_act; init)
	h_decoder_layers = [Dense(h_decoder[i] => h_decoder[i+1], h_act; init) for i in 1:length(h_decoder)-1]
	o_decoder_layers = Dense(h_decoder[end] => 2*h_representation; init)
	decoder = Chain(i_decoder_layer, h_decoder_layers..., o_decoder_layers)

    parallel_decoder = Chain(Dense(3=>4*h_representation,h_act),Dense(4*h_representation=>size_output(lt)))
	model = Chain(encoder_layer, decoder,parallel_decoder)
	return RnnTParallelModel(model, rng, sample)
end


function create_features(B::DualBundle, nn::RnnTParallelModel)
	sizeF = size_features(nn)
	ϕ = zeros(Float32, sizeF, size(B.w, 1))
	g = gS(B)
	gl = B.G[:, B.size]
	w = B.w
	z = zS(B)
	i = size_Bundle(B)
	ϕc = features_vector_i(B)
	for i in eachindex(g)
		ϕ[:, i] = [ϕc..., g[i], gl[i], w[i], z[i]]
	end
	return reshape(ϕ,(length(ϕ),1))
end

function size_features(_::RnnTParallelModel)
	return 27+4
end


function size_features(_::RnnTParallelModelfactory)
	return 27+4
end



function initializeϕ0(lt::AbstractModelFactory,ϕ::AbstractConcaveFunction)
    return []
end


function (m::RnnTParallelModel)(ϕ, B, ϵ = randn(B.nn.rng, Float32, 1))
	μ, σ2 = Flux.MLUtils.chunk(m.model[1:2](ϕ), 2, dims = 1)
	if m.sample
		σ2 = 2.0f0 .- softplus.(2.0f0 .- σ2)
		σ2 = -6.0f0 .+ softplus.(σ2 .+ 6.0f0)
		σ2 = exp.(σ2)
		sigma = sqrt.(σ2) / 100
		if m.train_mode
			ignore_derivatives() do
				push!(m.ϵs, ϵ)
			end
		end
		dev = (μ .+ ϵ .* sigma)
	else
		dev = μ
	end
	dev = mean(m.model[3](hcat(minimum(dev,dims=2),mean(dev,dims=2),maximum(dev,dims=2))'))
	t = m.deviation(B.params.t, dev)
	t = min.(B.params.t_max, abs.(t) .+ B.params.t_min)
	return t
end

Flux.@layer RnnTParallelModel

