"""
Factory used to create
"""
struct AttentionModelFactory <: AbstractDirectionAndTModelFactory

end


"""
Size of the features vector for each component of the Bundle.
It is associated to the last component added in the Bundle.
"""
function size_comp_features(lt::AttentionModelFactory)
	return 15
end

"""
Size of the features vector for the parameter t.
"""
function size_features(lt::AttentionModelFactory)
	return 42
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
	Ks::Zygote.Buffer{Float32, CUDA.CuArray{Float32, 3, CUDA.DeviceMemory}} # matrix og keys
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

	if !m.repeated
		m.Ks = Zygote.bufferfrom(device(zeros(Float32, bs ,  m.h_representation, it)))#[]
	end
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
	x = cat(xt,xγ;dims=1)#ndims(xt))

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

	
	
	if !m.repeated
		m.Ks[:, :,idx] = hk
		
		# compute the output to predict the new convex combination (after using it as input to a distribution function as softmax or sparsemax)
		aq = dropdims(hq,dims=2)
		ak = m.Ks[:, :, comps]
	
		# add the current hidden representation of the key to the matrix that store all the hidden representations
		
		γs = vcat([aq[:,i]'ak[i,:,:] for i in 1:size(ak,1)]...)
		return t, γs
	else
		# Assume hq and hk are CuArrays of shape (latent, iterations, batch)
		latent, T, B = size(hq)

	
		# Step 1: Extract hq at the last iteration → shape (latent, 1, batch)
		hq_last = hq[:, end:end, :]  # keep 3D shape: (latent, 1, B)

		# Step 2: Compute dot products across all hk iterations
		# We'll use batched_mul: (1, latent, B) × (latent, T, B) → (1, T, B)
		γs = batched_mul(permutedims(hq_last, (2,1,3)), hk)  # (1, T, B)

		# Step 3: Reshape to (T, B)
		γs = reshape(γs, B, T)

		return (device == gpu ? cu : identity)(reshape(t[:,end:end,:],1,:)), tanh.(γs) .* b
	end
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
	repeated::Bool = true
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
	model = AttentionModel(device(encoder), device(decoder_t),device(decoder_temperature), device(decoder_γk), device(decoder_γq), rng, sampling_t, sampling_θ, 1.0, h_representation, 1,h3_representations, repeated, device(Zygote.bufferfrom(device(zeros(Float32, bs , h_representation, it)))))
	return model
end

"""SoftBundle
Size of the hidden representation of an `AttentionModel`.
"""
function h_representation(nn::AttentionModel)
	return Int64(size(nn.decoder_t[1].weight, 2))
end



function create_features( B::SoftBundle, m::AttentionModel; auxiliary = 0)
    # We'll build two matrices: feat_t (global features) and feat_theta (per-vertex features)
	batch_size = length(B.idxComp) > 1 ? length(B.idxComp) : 1

	iters = m.repeated ? collect(1:B.size) : [B.li]
	#@show iters
	n_iter = m.repeated ? B.size : 1

    feat_t = zeros(Float32,size_features(B.lt),n_iter,batch_size)
    feat_theta = zeros(Float32,size_comp_features(B.lt),n_iter,batch_size)

    # Extract some constants once
    w = B.w
    zi = B.li
    si = B.s

    for (i, j) in enumerate(B.idxComp)
        # 1) scalar/time features
        ti = cpu(B.t)[i]
        # 2) objective and attention slices
        obj_i = cpu(B.obj[1,i, iters])
        α_i   = cpu(B.α[1,i, iters])
        θ_i   = cpu(reshape(B.θ[i, :], :))
        lp    = cpu(B.α)[1,i,1:length(θ_i)]' * θ_i

        # 3) pairwise products on z and G
        zz  = cpu(B.z[j,i, zi]' * B.z[j,i, iters])
        zsz = cpu(B.z[j,i, si[i]]' * B.z[j,i, iters])
        zszs = cpu(B.z[j,i, si[i]]' * B.z[j,i, si[i]])
        gg  = cpu(B.G[j,i, zi]' * B.G[j,i, iters])
        gsg = cpu(B.G[j,i, si[i]]' * B.G[j,i, iters])
		gsgs = cpu(B.G[j,i, si[i]]' * B.G[j,i, si[i]])
        ww  = (w[j,i]' * w[j,i])[1]   # scalar weight product

		ϕ = Float32[
            ti,
            ww,
            ti * ww,
            lp,
            ww > lp,
            ti*ww > lp,
            cpu(B.obj)[1,i,zi],
            cpu(B.obj)[1,i,si[i]],
            cpu(B.α)[1,i,zi],
            cpu(B.α)[1,i,si[i]],
            minimum(zz),   minimum(zsz),   minimum(gsg),   minimum(gg),  minimum(zszs),  minimum(gsgs),
            mean(zz),      mean(zsz),      mean(gsg),      mean(gg),  mean(zszs),  mean(gsgs),
            (length(zz)==1 ? 0f0 : std(zz)),
            (length(zsz)==1 ? 0f0 : std(zsz)),
            (length(gsg)==1 ? 0f0 : std(gsg)),
            (length(gg)==1   ? 0f0 : std(gg)),
            (length(zszs)==1 ? 0f0 : std(zszs)),
            (length(gsgs)==1 ? 0f0 : std(gsgs)),
            maximum(zz),   maximum(zsz),  maximum(zszs),  maximum(gsgs),  maximum(gsg),   maximum(gg),
            (sum(zz))/2,
            (sum(zsz))/2,
            (sum(gsg))/2,
            (sum(gg))/2,  
            (sum(zszs))/2,
            (sum(gsgs))/2,      
			sum(B.G[j,i, B.li]'*w[j,i]),
        	sum(B.G[j,i, si[i]]'*w[j,i]),
        ]

        # 5) Build the per-vertex feature vector ϕγ (Float32)
        z_seg = B.z[j,i, iters]
        G_seg = B.G[j,i, iters]
		
		ϕγ = hcat(cpu([
			mean(G_seg,dims=1)', std(G_seg,dims=1)', minimum(G_seg,dims=1)', maximum(G_seg,dims=1)',
            mean(z_seg,dims=1)', std(z_seg,dims=1)', minimum(z_seg,dims=1)', maximum(z_seg,dims=1)',
            B.G[j,i, iters]'*B.w[j,i],
            α_i,
            obj_i,
            obj_i .< cpu(B.obj)[1,i,B.li],
            obj_i .< cpu(B.obj)[1,i,si[i]],
            B.li .== collect(iters),
            si[i] .== collect(iters)])...)
      
        # 6) Stack into matrices
        feat_t[:,:,i]     .= ϕ
        feat_theta[:,:,i] = ϕγ'
        
    end

    # return in the expected shape: (batch_size × n_features)
    return device(feat_t), device(feat_theta)
end
