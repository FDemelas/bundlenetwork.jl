using Documenter

makedocs(
    sitename = "BundleNetworks Documentation",
    format = Documenter.HTML(
        prettyurls = false,  # Important for GitLab Pages!
        edit_link = "main",
    ),
    pages = [
        "Home" => "index.md",
        "Installation" => "installation.md",
        "Quick Start" => "quickstart.md",
        "Tutorials" => [
            "Bundle Networks" => "tutorials/bundle_networks.md",
            "Hyper Parameters Learning" => "tutorials/hyper_parameters.md",
            "Baselines" => "tutorials/baselines.md",
        ],
        "Manual" => [
            "Architecture" => "manual/architecture.md",
        ]
    ],
    authors = "Francesco Demelas",
)

# NO deploydocs() for GitLab Pages - CI handles it automatically
