
using DrWatson
@quickactivate "."

using AbstractMCMC
using DelimitedFiles
using Distributions
using DrWatson
using JLD2
using MCMCChains
using MCMCDiagnosticTools
using Random, Random123
using ReversibleJump
using SinusoidDetection
using Statistics

# function setup()
#     R"options(download.file.method=\"wget\");" *
#         "devtools::install_github(\"hsong1/momentLS\", ref=\"main\")"
# end

# function estimate_autocovariance(x)
#     r           = autocov(x)
#     delta       = tune_delta(x, nSplits = 5, c_M_const = 0)[:delta]
#     delta_tilde = R"$delta*0.8"
#     m           = SR1(r, delta = delta_tilde)

#     asympVariance(weights = m[:weights], support = m[:support])
#     estimated_autocov = computeMoments(support = m$support, weights = m$weights, M = 100)
# end

mutable struct TrackedModel{
    model <: SinusoidDetection.AbstractSinusoidModel
} <: SinusoidDetection.AbstractSinusoidModel
    evals
    model
end

function run_sampling(
    rng        ::Random.AbstractRNG,
    model      ::SinusoidDetection.AbstractSinusoidModel,
    rjmcmc     ::ReversibleJump.AbstractRJMCMCSampler,
    n_burn     ::Int,
    n_samples  ::Int,
    init_delta2::Real
)
    initial_params = [init_delta2]
    initial_order  = 0
    samples, stats = ReversibleJump.sample(
        rjmcmc,
        model,
        n_samples + n_burn,
        initial_order,
        initial_params,
        show_progress=true
    )
    order_chain = [stat.order for stat in stats]
    last(order_chain, n_samples)
end

function run_config(rng, config)
    N  = 64
    Δω = π/15
    σ² = 3.0
    ω  = (0.5 .+ Δω*(0:config.ntargets-1)) |> collect
    δ² = 10^(config.snr/10)
    y  = SinusoidDetection.sample_signal(rng, ω, N, σ², δ²)

    n_burn    = 10
    n_samples = 100
    n_anneal  = 4

    ν0    = 0.0
    γ0    = 0.0
    α_δ²  = 1/2
    β_δ²  = N/2
    α_λ   = 0.1
    β_λ   = 0.1 
    k_max = floor(Int, (N-1)/2)
    prior = truncated(NegativeBinomial(1/2 + α_λ, β_λ/(β_λ+1)), upper=k_max)
    prop  = SinusoidLocalProposal()
    path  = ArithmeticPath()

    model, init_delta2, jump, mcmc = if config.mcmc == :indep_imhrwmh
        #
        model = SinusoidUnknownSNR(y, ν0, γ0, α_δ², β_δ², prior)
	mcmc  = IMHRWMHSinusoid(model)
        jump  = IndepJumpProposal(prop)
        model, 1.0, jump, mcmc
    elseif config.mcmc == :indep_slice
        #
        model = SinusoidUnknownSNR(y, ν0, γ0, α_δ², β_δ², prior)
        mcmc  = SliceSinusoid(SliceSteppingOut(), model, 1.0, 1.0)
        jump  = IndepJumpProposal(prop)
        model, 1.0, jump, mcmc
    elseif config.mcmc == :indep_slice_reparam
        #
        model = SinusoidUnknownSNRReparam(y, ν0, γ0, α_δ², β_δ², prior)
        mcmc  = SliceSinusoid(SliceSteppingOut(), model, 1.0, 1.0)
        jump  = IndepJumpProposal(prop)
        model, 0.0, jump, mcmc
    elseif config.mcmc == :ais_slice
        #
        model = SinusoidUnknownSNR(y, ν0, γ0, α_δ², β_δ², prior)
        mcmc  = SliceSinusoid(SliceSteppingOut(), model, 1.0, 1.0)
	jump  = AnnealedJumpProposal(n_anneal, prop, path)
        model, 1.0, jump, mcmc
    elseif config.mcmc == :ais_slice_reparam
        #
        model = SinusoidUnknownSNRReparam(y, ν0, γ0, α_δ², β_δ², prior)
        mcmc  = SliceSinusoid(SliceSteppingOut(), model, 1.0, 1.0)
	jump  = AnnealedJumpProposal(n_anneal, prop, path)
        model, 0.0, jump, mcmc
    end

    rjmcmc = if config.rjmcmc == :rjmcmc
        ReversibleJump.ReversibleJumpMCMC(prior, jump, mcmc)
    else
        ReversibleJump.NonReversibleJumpMCMC(jump, mcmc; jump_rate=0.8)
    end
    
    samples = run_sampling(rng, model, rjmcmc, n_burn, n_samples, init_delta2)
    Dict("samples" => samples)
end

function main()
    key  = 1
    seed = (0x38bef07cf9cc549d, 0x49e2430080b3f797)
    rng  = Philox4x(UInt64, seed, 8)
    Random123.set_counter!(rng, key)

    # SNR         : 0dB, 5dB, 10dB
    # num targets : 1, 5, 10

    snrs      = [(snr       = snr,)      for snr       in [0, 5, 10]]
    ntargets  = [(ntargets  = ntargets,) for ntargets  in [1, 5, 10]]
    rjmcmcs   = [(rjmcmc = rjmcmc,)      for rjmcmc    in [:rjmcmc, :nrjmcmc]]
    mcmcs     = [
        (mcmc = :indep_imhrwmh,),
        (mcmc = :indep_slice,),
        (mcmc = :indep_slice_reparam,),
        (mcmc = :ais_slice,),
        (mcmc = :ais_slice_reparam,),
    ]

    configs = Iterators.product(snrs, ntargets, rjmcmcs, mcmcs) |> collect
    configs = reshape(configs, :)
    configs = map(x -> merge(x...), configs)

    for config in configs
        fname = savename("pre", config)
        DrWatson.produce_or_load(config, datadir("exp_raw"), prefix="longchain") do config′
            run_config(rng, config′)
        end
    end
end
