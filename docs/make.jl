using SinusoidDetection
using Documenter

DocMeta.setdocmeta!(SinusoidDetection, :DocTestSetup, :(using SinusoidDetection); recursive=true)

makedocs(;
    modules=[SinusoidDetection],
    authors="Kyurae Kim <kyrkim@seas.upenn.edu> and contributors",
    repo="https://github.com/Red-Portal/SinusoidDetection.jl/blob/{commit}{path}#{line}",
    sitename="SinusoidDetection.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://Red-Portal.github.io/SinusoidDetection.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/Red-Portal/SinusoidDetection.jl",
    devbranch="main",
)
