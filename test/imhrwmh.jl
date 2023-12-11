
function MCMCTesting.markovchain_transition(
    rng  ::Random.AbstractRNG,
    model::SinusoidFixedOrderModel{<:SinusoidKnownSNR},
    mcmc ::IMHRWMHSinusoid{<:SinusoidKnownSNR},
    θ, y
)
    model_base = model.model
    model_base = @set model_base.y = y
    model_base = @set model_base.freqprop = Uniform(0, π)
    ReversibleJump.transition_mcmc(rng, mcmc, model_base, copy(θ)) |> first
end

function MCMCTesting.markovchain_transition(
    rng  ::Random.AbstractRNG,
    model::SinusoidFixedOrderModel{<:SinusoidUnknownSNR},
    mcmc ::IMHRWMHSinusoid{<:SinusoidUnknownSNR},
    θ, y
)
    model_base = model.model
    model_base = @set model_base.y = y
    model_base = @set model_base.freqprop = Uniform(0, π)
    ReversibleJump.transition_mcmc(rng, mcmc, model_base, copy(θ)) |> first
end

@testset "imhrwmh known snr" begin
    nu0    = 2.0
    gamma0 = 5.0
    delta2 = 8.0
    N      = 16
    k      = 1
    prior  = truncated(Poisson(3), upper=floor(Int, (N-1)/2))

    model_base = rand_sinusoids_knownsnr(N, nu0, gamma0, delta2, prior, k)
    mcmc       = IMHRWMHSinusoid(model_base)
    model      = SinusoidFixedOrderModel(k, model_base)
    @testset "determinism" begin
        n_mcmc_steps = 10
        θ_init, _ = MCMCTesting.sample_joint(Random.default_rng(), model)

        rng = StableRNG(1)
        θ   = copy(θ_init)
        for i in 1:n_mcmc_steps
           θ, _ = ReversibleJump.transition_mcmc(rng, mcmc, model_base, θ)
        end

        rng = StableRNG(1)
        θ′  = copy(θ_init)
        for i in 1:n_mcmc_steps
           θ′, _ = ReversibleJump.transition_mcmc(rng, mcmc, model_base, θ′)
        end
        @test θ == θ′
    end

    @testset "inference" begin
        _, y  = MCMCTesting.sample_joint(Random.default_rng(), model)
        model′ = @set model.model.y = y

        n_pvalue_samples = 32
        n_samples        = 1000
        n_mcmc_steps     = 10
        test             = TwoSampleTest(n_samples, n_mcmc_steps)

        subject = TestSubject(model′, mcmc)
        @test seqmcmctest(test, subject, 0.0001, n_pvalue_samples; show_progress=true)
    end
end

@testset "imhrwmh unknown snr" begin
    ν0 = 2.0
    γ0 = 5.0
    N  = 16

    # Due to numerical accuracy issues, the prior needs to be set carefully
    α_δ² = 40.0
    β_δ² = 5.0

    k          = 1
    prior      = truncated(Poisson(3), upper=floor(Int, (N-1)/2))
    model_base = rand_sinusoids_unknownsnr(N, ν0, γ0, α_δ², β_δ², prior, k)
    model      = SinusoidFixedOrderModel(k, model_base)
    mcmc       = IMHRWMHSinusoid(model_base)

    @testset "determinism" begin
        n_mcmc_steps = 10
        θ_init, y = MCMCTesting.sample_joint(Random.default_rng(), model)

        rng = StableRNG(1)
        θ   = copy(θ_init)
        for i in 1:n_mcmc_steps
           θ, _ = ReversibleJump.transition_mcmc(rng, mcmc, model_base, θ)
        end

        rng = StableRNG(1)
        θ′  = copy(θ_init)
        for i in 1:n_mcmc_steps
           θ′, _ = ReversibleJump.transition_mcmc(rng, mcmc, model_base, θ′)
        end
        @test θ == θ′
    end

    @testset "inference" begin
        _, y  = MCMCTesting.sample_joint(Random.default_rng(), model)
        model′ = @set model.model.y = y

        n_pvalue_samples = 32
        n_samples        = 1000
        n_mcmc_steps     = 10
        test             = TwoSampleTest(n_samples, n_mcmc_steps)

        subject = TestSubject(model′, mcmc)
        @test seqmcmctest(test, subject, 0.0001, n_pvalue_samples; show_progress=true)
    end
end
