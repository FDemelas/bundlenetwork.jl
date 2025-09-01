
"""
Factory used to create
"""
struct AttentionModelFactory <: AbstractDirectionAndTModelFactory

end

"""
Create the features vector for a model of type `AttentionModelFactory` using the informations contained in the bundle `B`.
`B` should be of type `SoftBundle`.
"""
function create_features(lt::AttentionModelFactory, B::SoftBundle; auxiliary = 0)
	t = sum(B.t)
	obj = cpu(B.obj)
	θ = cpu(B.θ)[1:end]
	α = cpu(B.α[1:B.size])
	i, s, e = 1, 1, length(B.w)
	lp = sum(α[1:length(θ)]' * θ)
	qp = sum(B.w' * B.w)

	zz = cpu(B.z[:, :]' * B.z[:, B.size])
	zsz = cpu(B.z[:, :]' * B.z[:, B.s])
	gg = cpu(B.G[:, :]' * B.G[:, B.size])
	gsg = cpu(B.G[:, :]' * B.G[:, B.s])

	ϕ = Float32[t,
		qp,
		t*qp,
		lp,
		qp>lp,
		10000*qp>lp,
		obj[B.li],
		obj[B.s],
		obj[B.li]<obj[B.s],
		B.li,
		α[B.s],
		α[B.li],
		sum(sqrt(zz[B.li]) / 2),
		sum(sqrt(zsz[B.s]) / 2),
		sum(sqrt(gsg[B.s]) / 2),
		sum(sqrt(gg[B.li]) / 2),
	]

	ϕγ = Float32[mean(B.G[:, B.li]),
		mean(B.z[:, B.li]),
		std(B.G[:, B.li]),
		std(B.z[:, B.li]),
		minimum(B.G[:, B.li]),
		minimum(B.z[:, B.li]),
		maximum(B.G[:, B.li]),
		maximum(B.z[:, B.li]),
		minimum(zz),
		minimum(zsz),
		minimum(gsg),
		minimum(gg),
		B.G[:, B.li]'*B.w,
		maximum(zz),
		maximum(zsz),
		maximum(gsg),
		maximum(gg),
		B.G[:, B.s]'*B.w,
		α[B.li],
		obj[B.li]]
	return device(ϕ), device(ϕγ)
end



"""
Create the features vector for a model of type `AttentionModelFactory` using the informations contained in the bundle `B`.
`B` should be of type `BatchedSoftBundle`.
"""
function create_features(lt::AttentionModelFactory, B::BatchedSoftBundle; auxiliary = 0)
	let feat_t, feat_theta
		t = B.t
		mli = 1
		obj = B.obj[:, 1:B.size]
		α = B.α[:, 1:B.size]
		for (i, (s, e)) in enumerate(B.idxComp)
			θ = cpu(reshape(B.θ[i, :], :))
			lp = α[i, 1:length(θ)]' * θ

			zz = cpu(B.z[s:e, B.li]' * B.z[s:e, mli:B.size])
			zsz = cpu(B.z[s:e, B.s[i]]' * B.z[s:e, mli:B.size])
			gg = cpu(B.G[s:e, B.li]' * B.G[s:e, mli:B.size])
			gsg = cpu(B.G[s:e, B.s[i]]' * B.G[s:e, mli:B.size])
			ww = B.w[s:e]' * B.w[s:e]

			ϕ = Float32[t[i],
				ww,
				t[i]*ww,
				lp,
				ww>lp,
				10000*ww>lp,
				obj[i, B.li],
				obj[i, B.s[i]],
				obj[i, B.li]<obj[i, B.s[i]],
				B.li,
				α[i, B.s[i]],
				α[i, B.li],
				sqrt(sum(zz[B.li]))/2,
				sqrt(sum(zsz[B.s[i]]) / 2),
				sqrt(sum(gsg[B.s[i]]) / 2),
				sqrt(sum(gg[B.li]) / 2),
			]
			ϕγ = Float32[mean(B.G[s:e, B.li]),
				mean(B.z[s:e, B.li]),
				std(B.G[s:e, B.li]),
				std(B.z[s:e, B.li]),
				minimum(B.G[s:e, B.li]),
				minimum(B.z[s:e, B.li]),
				maximum(B.G[s:e, B.li]),
				maximum(B.z[s:e, B.li]),
				minimum(zz),
				minimum(zsz),
				minimum(gsg),
				minimum(gg),
				B.G[s:e, B.li]'*B.w[s:e],
				maximum(zz),
				maximum(zsz),
				maximum(gsg),
				maximum(gg),
				B.G[s:e, B.s[i]]'*B.w[s:e],
				α[i, B.li],
				obj[i, B.li]]
			if i == 1
				feat_t, feat_theta = ϕ, ϕγ
			else
				feat_theta = hcat(feat_theta, ϕγ)
				feat_t = hcat(feat_t, ϕ)
			end

		end
		return feat_t, feat_theta
	end
end

"""
Size of the features vector for each component of the Bundle.
It is associated to the last component added in the Bundle.
"""
function size_comp_features(lt::AttentionModelFactory)
	return 20
end

"""
Size of the features vector for the parameter t.
"""
function size_features(lt::AttentionModelFactory)
	return 16
end


mutable struct AttentionModel <: AbstractModel
	encoder::Chain # model to encode mean and variance to sample hidden representations
	decoder_t::Chain # decode t from hidden representation
	decoder_temperature::Chain # decode t from hidden representation
	decoder_γk::Chain# decode keys from hidden representation
	decoder_γq::Chain# decode query from hidden representation
	rng::MersenneTwister #random number generator
	sample_t::Bool # if true sample hidden representation of t, otherwhise take the mean (i.e. no sampling)
	sample_γ::Bool# if true sample hidden representation of keys and queries, otherwhise take the mean (i.e. no sampling)
	rescaling_factor::Int64 # rescaling_factor for the output γs (unused for the moment)
	h_representation::Int64 # size of hidden representations
	it::Int64 # iteration counter
	h3_representations::Bool # if true use three indipendent hidden representations instead of one
	repeated::Bool # if true recompute the inputs for all the previous iterations
	Ks::Zygote.Buffer{Float32, CUDA.CuArray{Float32, 2, CUDA.DeviceMemory}} # matrix og keys
	use_tanh::Bool
end

"""
Reset the networks composing an `AttentionModel` and re-initialize the matrix `Ks` used to stock the keys of the bundle components.
Note it needs not only the model `m::AttentionModel`, but also two additional parameter `bs` and `it` refering to the batch size that will be considered as input to the model and the maximum number of iterations in the Bundle.
In practice `bs` should be exactly the batch-size (by default `1`), while `it` can be also grater than the maximum number of iterations (by default `500`).
"""
function reset!(m::AttentionModel, bs::Int = 1, it::Int = 500)
	# reset the model component to forgot history in RNN
	Flux.reset!(m.encoder)
	Flux.reset!(m.decoder_t)
	Flux.reset!(m.decoder_temperature)
	Flux.reset!(m.decoder_γk)
	Flux.reset!(m.decoder_γq)

#	if !m.repeated
		m.Ks = Zygote.bufferfrom(device(zeros(Float32, bs * m.h_representation, it)))#[]
#	end
	# reset iteration counter to 1
	m.it = 1
end

"""
Forward computation of an `AttentionModel`. The Backward is made by automatic differentiation.
The inputs are:
- `xt`: features of t;
- `xγ`: features of the bundle component.
"""
function (m::AttentionModel)(xt, xγ, idx, comps)
	# append the features for t and for γs
	x = vcat(xt,xγ)#ndims(xt))

	# encode the full hidden representation
	h = m.encoder(x)
	# divide the hidden representation in mean and variance for t,the query and the key
	μ, σ2 = Flux.MLUtils.chunk(h, 2, dims = 1)

	#construct the values to sample the hidden representation of t 
	σ2 = 2.0f0 .- softplus.(2.0f0 .- σ2)
	σ2 = -6.0f0 .+ softplus.(σ2 .+ 6.0f0)
	σ2 = exp.(σ2)
	sigma = sqrt.(σ2 .+ 1)

	μt,μb, μk,μq = m.h3_representations ? copy.(Flux.MLUtils.chunk(μ, 4, dims = 1)) : (copy(μ) ,copy(μ) ,copy(μ),copy(μ))
	σt,σb,σk,σq = m.h3_representations ? Flux.MLUtils.chunk(sigma, 4, dims = 1) : (sigma,sigma,sigma,sigma)

	# create the random component for the sample of the time
	ϵt,ϵb,ϵk,ϵq =  device(randn(m.rng, Float32, size(μt))), device(randn(m.rng, Float32, size(μb))), device(randn(m.rng, Float32, size(μk))), device(randn(m.rng, Float32, size(μq)))
	
	# sample the hidden representation of t and then give it as input to the decoder to compute t
	t = m.decoder_t(m.sample_t ? μt .+ ϵt .* σt : μt )

	b = m.decoder_temperature(m.sample_t ? μb .+ ϵb .* σb : μb )

	# sample the hidden representation of the key and the query
	hk = m.decoder_γk(m.sample_γ ? μk .+ ϵk .* σk : μk )
	hq = m.decoder_γq(m.sample_γ ? μq .+ ϵq .* σq : μq )

#	@show size(m.Ks[:,idx])
	m.Ks[:, idx] = reshape(hk,:)
#		
		# compute the output to predict the new convex combination (after using it as input to a distribution function as softmax or sparsemax)
#		aq = dropdims(hq,dims=2)
		aq = (size(hq, 2) > 1 ? Flux.MLUtils.chunk(hq, size(hq, 2); dims = 2) : [hq])
		ak = (Flux.MLUtils.chunk(m.Ks[:, comps], Int64(size(m.Ks, 1) / m.h_representation); dims = 1))

		#ak = m.Ks[:, comps]
	
		# add the current hidden representation of the key to the matrix that store all the hidden representations
		
#		γs = vcat([aq[:,i]'ak[i,:] for i in 1:size(ak,1)]...)
		γs = vcat(map((x, y) -> sum(x'y; dims = 1), aq, ak)...)

		return t, γs
end

# define `AttentionModel` as a Flux layer
Flux.@layer AttentionModel

"""
Function to create an `AttentionModel` given its hyper-parameters.
"""
function create_NN(
	lt::AttentionModelFactory;
	h_act = gelu,
	h_representation::Int = 128,
	h_decoder::Vector{Int} = [h_representation * 8],
	seed::Int = 1,
	norm::Bool = false,
	sampling_t::Bool = false,
	sampling_θ::Bool = true,
	ot_act = softplus,
	rnn = true,
	h3_representations::Bool = false,
	repeated::Bool = true,
	use_tanh::Bool = false
)
	bs,it=1,1
	# possibly a normalization function, but is `norm` is false, then it is the identity
	f_norm(x) = norm ? Flux.normalise(x) : identity(x)

	#construct layer initilizer (for layer parameters)
	rng = MersenneTwister(seed)
	init = Flux.truncated_normal(Flux.MersenneTwister(seed); mean = 0.0, std = 0.01)

	#the encoder used to predict the hidden space, given the features of the current iteration,
	encoder = rnn ? Chain(f_norm, LSTM(size_features(lt) + size_comp_features(lt) => (h3_representations ? 8 : 2) * h_representation)) : Chain(f_norm, Dense(size_features(lt) + size_comp_features(lt) => (h3_representations ? 8 : 2) * h_representation,h_act))

	# construct the decoder that predicts `t` from its hidden representation
	i_decoder_layer = Dense(h_representation => h_decoder[1], h_act; init)
	h_decoder_layers = [Dense(h_decoder[i] => h_decoder[i+1], h_act; init) for i in 1:length(h_decoder)-1]
	o_decoder_layers = Dense(h_decoder[end] => 1, ot_act; init)
	decoder_t = Chain(i_decoder_layer, h_decoder_layers..., o_decoder_layers)

	# construct the decoder that predicts `t` from its hidden representation
	i_decoder_layer = Dense(h_representation => h_decoder[1], h_act; init)
	h_decoder_layers = [Dense(h_decoder[i] => h_decoder[i+1], h_act; init) for i in 1:length(h_decoder)-1]
	o_decoder_layers = Dense(h_decoder[end] => 1, ot_act; init)
	decoder_temperature = Chain(i_decoder_layer, h_decoder_layers..., o_decoder_layers)

	# construct the decoder that predicts the query from its hidden representation
	i_decoder_layer = Dense(h_representation => h_decoder[1], h_act; init)
	h_decoder_layers = [Dense(h_decoder[i] => h_decoder[i+1], h_act; init) for i in 1:length(h_decoder)-1]
	o_decoder_layers = Dense(h_decoder[end] => h_representation; init)
	decoder_γq = Chain(i_decoder_layer, h_decoder_layers..., o_decoder_layers)

	# construct the decoder that predicts the key from its hidden representation
	i_decoder_layer = Dense(h_representation => h_decoder[1], h_act; init)
	h_decoder_layers = [Dense(h_decoder[i] => h_decoder[i+1], h_act; init) for i in 1:length(h_decoder)-1]
	o_decoder_layers = Dense(h_decoder[end] => h_representation; init)
	decoder_γk = Chain(i_decoder_layer, h_decoder_layers..., o_decoder_layers)

	#put all together to construct an `AttentionModel`
	model = AttentionModel(device(encoder), device(decoder_t),device(decoder_temperature), device(decoder_γk), device(decoder_γq), rng, sampling_t, sampling_θ, 1.0, h_representation, 1,h3_representations, repeated, device(Zygote.bufferfrom(device(zeros(Float32, bs* h_representation, it)))),use_tanh)
	return model
end

"""SoftBundle
Size of the hidden representation of an `AttentionModel`.
"""
function h_representation(nn::AttentionModel)
	return Int64(size(nn.decoder_t[1].weight, 2))
end
