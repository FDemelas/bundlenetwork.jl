"""
Factory used to create
"""
struct AttentionModelFactory <: AbstractDirectionAndTModelFactory end

"""
Create the features vector for a model of type `AttentionModelFactory` using the informations contained in the bundle `B`.
`B` should be of type `SoftBundle`.
"""
function create_features(lt::AttentionModelFactory, B::SoftBundle; auxiliary = 0)
	t = sum(B.t)
	obj = cpu(B.obj)
	jθ = B.li == 1 ? 1 : B.li - 1
	θ = cpu(B.θ)[1:jθ]
	α = cpu(B.α[1:B.li])
	i, s, e = 1, 1, length(B.w)
	lp = sum(α[1:length(θ)]' * θ)
	qp = sum(B.w' * B.w)

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
		sum(sqrt(B.z[s:e, B.li]' * B.z[s:e, B.li]) / 2),
		sum(sqrt(B.z[s:e, B.s]' * B.z[s:e, B.s]) / 2),
		sum(sqrt(B.G[s:e, B.s]' * B.G[s:e, B.s]) / 2),
		sum(sqrt(B.G[s:e, B.li]' * B.G[s:e, B.li]) / 2),
	]

	ϕγ = Float32[mean(B.G[s:e, B.li]),
		mean(B.z[s:e, B.li]),
		std(B.G[s:e, B.li]),
		std(B.z[s:e, B.li]),
		minimum(B.G[s:e, B.li]),
		minimum(B.z[s:e, B.li]),
		maximum(B.G[s:e, B.li]),
		maximum(B.z[s:e, B.li]),
		minimum(B.z[s:e, B.li]' * B.z[s:e, j] for j in 1:B.li),
		minimum(B.z[s:e, B.s]' * B.z[s:e, j] for j in 1:B.li),
		minimum(B.G[s:e, B.s]' * B.G[s:e, j] for j in 1:B.li),
		minimum(B.G[s:e, B.li]' * B.G[s:e, j] for j in 1:B.li),
		B.G[s:e, B.li]'*B.w[s:e],
		maximum(B.z[s:e, B.li]' * B.z[s:e, j] for j in 1:B.li),
		maximum(B.z[s:e, B.s]' * B.z[s:e, j] for j in 1:B.li),
		maximum(B.G[s:e, B.s]' * B.G[s:e, j] for j in 1:B.li),
		maximum(B.G[s:e, B.li]' * B.G[s:e, j] for j in 1:B.li),
		B.G[s:e, B.s]'*B.w[s:e],
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
		mli = 1#max(1, B.li)# - 99)
		obj = B.obj[:, 1:B.li]
		α = B.α[:, 1:B.li]
		for (i, (s, e)) in enumerate(B.idxComp)
			θ = cpu(reshape(B.θ[i, :], :))
			lp = α[i, 1:max(1, B.li - 1)]' * θ

			zz = cpu(B.z[s:e, B.li]' * B.z[s:e, mli:B.li])#[B.z[s:e, B.li]' * B.z[s:e, j] for j in mli:B.li]
			zsz = cpu(B.z[s:e, B.s[i]]' * B.z[s:e, mli:B.li])
			gg = cpu(B.G[s:e, B.li]' * B.G[s:e, mli:B.li])
			gsg = cpu(B.G[s:e, B.s[i]]' * B.G[s:e, mli:B.li])
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
				sqrt(sum(zz[:, B.li]))/2,
				sqrt(sum(zsz[:, B.s[i]]) / 2),
				sqrt(sum(gsg[:, B.s[i]]) / 2),
				sqrt(sum(gg[:, B.li]) / 2),
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
			#            ϕγ = vcat(α[i, B.li],obj[i, B.li])#B.z[idxs, B.li],B.G[idxs, B.li],

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
	encoder_t::Any
	decoder_t::Any
	encoder_γ::Any
	decoder_γk::Any
	decoder_γq::Any
	rng::Any
	sample_t::Bool
	sample_γ::Bool
	Ks::Any
	μs::Any
	σs::Any
	ϵs::Any
	rescaling_factor::Int64
	log_trick::Bool
	h_representation::Int64
	it::Int64
end

"""
Reset the networks composing an `AttentionModel` and re-initialize the matrix `Ks` used to stock the keys of the bundle components.
Note it needs not only the model `m::AttentionModel`, but also two additional parameter `bs` and `it` refering to the batch size that will be considered as input to the model and the maximum number of iterations in the Bundle.
In practice `bs` should be exactly the batch-size (by default `1`), while `it` can be also grater than the maximum number of iterations (by default `500`).
"""
function reset!(m::AttentionModel, bs::Int = 1, it::Int = 500)
	Flux.reset!(m.encoder_t)
	Flux.reset!(m.encoder_γ)
	Flux.reset!(m.decoder_t)
	Flux.reset!(m.decoder_γk)
	Flux.reset!(m.decoder_γq)

	m.Ks = Zygote.bufferfrom(device(zeros(Float32, bs * m.h_representation, it)))
	if m.log_trick
		m.μs = Zygote.bufferfrom(device(zeros(Float32, 2 * m.h_representation * bs, it)))
		m.σs = Zygote.bufferfrom(device(zeros(Float32, 2 * m.h_representation * bs, it)))
		m.ϵs = Zygote.bufferfrom(device(zeros(Float32, 2 * m.h_representation * bs, it)))
	else
		m.μs = []
		m.σs = []
		m.ϵs = []
	end

	m.it = 1
end

function (m::AttentionModel)(xt, xγ)
	h = m.encoder_t(xt)

	μ_hki, σ_hki, μ_hqi, σ_hqi = Flux.MLUtils.chunk(m.encoder_γ(xγ), 4, dims = 1)

	μ, σ2 = Flux.MLUtils.chunk(m.decoder_t(h), 2, dims = 1)
	σ2 = 2.0f0 .- softplus.(2.0f0 .- σ2)
	σ2 = -6.0f0 .+ softplus.(σ2 .+ 6.0f0)
	σ2 = exp.(σ2)
	sigma = sqrt.(σ2 .+ 1)
	ϵ = randn(m.rng, Float32, size(μ))

	t = m.sample_t ? abs.(μ .+ ϵ .* sigma) : abs.(μ)

	ϵk = device(randn(m.rng, Float32, size(μ_hki)))
	ϵq = device(randn(m.rng, Float32, size(μ_hqi)))


	σ2 = 2.0f0 .- softplus.(2.0f0 .- σ_hqi)
	σ2 = -6.0f0 .+ softplus.(σ2 .+ 6.0f0)
	σ2 = exp.(σ2)
	σ_hqi = sqrt.(σ2 .+ 1)

	σ2 = 2.0f0 .- softplus.(2.0f0 .- σ_hki)
	σ2 = -6.0f0 .+ softplus.(σ2 .+ 6.0f0)
	σ2 = exp.(σ2)
	σ_hki = sqrt.(σ2 .+ 1)

	hki = m.decoder_γk(m.sample_γ ? μ_hki + ϵk .* σ_hki : μ_hki)
	hqi = m.decoder_γq(m.sample_γ ? μ_hqi + ϵq .* σ_hqi : μ_hqi)

	m.Ks[:, m.it] = reshape(hki, :)

	if m.log_trick
		m.μs[:, m.it] = vcat(μ_hki, μ_hqi)
		m.ϵs[:, m.it] = vcat(ϵk, ϵq)
		m.σs[:, m.it] = vcat(σ_hki, σ_hqi)
	end
	aq = device(size(hqi, 2) > 1 ? Flux.MLUtils.chunk(hqi, size(hqi, 2); dims = 2) : [hqi])
	ak = (Flux.MLUtils.chunk(m.Ks[:, 1:m.it], Int64(size(m.Ks, 1) / m.h_representation); dims = 1))

	θg = vcat(map((x, y) -> sum(x'y; dims = 1), aq, ak)...)
	m.it = m.it + 1
	return t, θg
end

Flux.@layer AttentionModel

function create_NN(lt::AttentionModelFactory; h_act = gelu, h_representation::Int = 32, h_decoder::Vector{Int} = [h_representation * 8], seed::Int = 1, norm::Bool = false, sampling_t::Bool = false, sampling_θ::Bool = true, log_trick::Bool = false)
	f_norm(x) = norm ? Flux.normalise(x) : identity(x)
	rng = MersenneTwister(seed)
	init = Flux.truncated_normal(Flux.MersenneTwister(seed); mean = 0.0, std = 0.01)
	encoder_t = Chain(f_norm, LSTM(size_features(lt) => h_representation))

	i_decoder_layer = Dense(h_representation => h_decoder[1], h_act; init)
	h_decoder_layers = [Dense(h_decoder[i] => h_decoder[i+1], h_act; init) for i in 1:length(h_decoder)-1]
	o_decoder_layers = Dense(h_decoder[end] => 2; init)
	decoder_t = Chain(i_decoder_layer, h_decoder_layers..., o_decoder_layers)

	encoder_γ = Chain(f_norm, LSTM(size_comp_features(lt) => 4 * h_representation))

	i_decoder_layer = Dense(h_representation => h_decoder[1], h_act; init)
	h_decoder_layers = [Dense(h_decoder[i] => h_decoder[i+1], h_act; init) for i in 1:length(h_decoder)-1]
	o_decoder_layers = Dense(h_decoder[end] => h_representation; init)
	decoder_γq = Chain(i_decoder_layer, h_decoder_layers..., o_decoder_layers)

	i_decoder_layer = Dense(h_representation => h_decoder[1], h_act; init)
	h_decoder_layers = [Dense(h_decoder[i] => h_decoder[i+1], h_act; init) for i in 1:length(h_decoder)-1]
	o_decoder_layers = Dense(h_decoder[end] => h_representation; init)
	decoder_γk = Chain(i_decoder_layer, h_decoder_layers..., o_decoder_layers)

	model = AttentionModel(device(encoder_t), device(decoder_t), device(encoder_γ), device(decoder_γk), device(decoder_γq), rng, sampling_t, sampling_θ, device([]), device([]), device([]), device([]), 1.0, log_trick, h_representation, 1)
	return model
end

function h_representation(nn::AttentionModel)
	return Int64(size(nn.decoder_t[1].weight, 2))
end
