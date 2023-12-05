
using Revise

using Distributions
using SinusoidDetection
using ReversibleJump
using Random
using MCMCChains
using MCMCDiagnostics
using DelimitedFiles
using Statistics
using SignalAnalysis

function 

function main(rng::AbstractRNG)
    k  = rand(rng, orderprior)
    ω  = rand(rng, Uniform(0, π), k)
    σ² = rand(rng, InverseGamma(nu0/2, gamma0/2))
    y  = SinusoidDetection.sample_signal(rng, ω, N, σ², delta2)
end
