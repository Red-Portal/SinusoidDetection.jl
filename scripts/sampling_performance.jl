
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

function main()
    
rand_sinusoids_unknownsnr(
    rng, N, nu0, gamma0, alpha_delta2, beta_delta2,
    orderprior = truncated(Poisson(3), upper=floor(Int, (N-1)/2))
)
end
