using Documenter

makedocs(
    sitename = "BundleNetworks Documentation",
    format = Documenter.HTML(
        prettyurls = false,  # Set to false for GitLab Pages
        canonical = "https://depot.lipn.univ-paris13.fr/demelas/bundlenetwork.jl",
        assets = String["assets/custom.css"],
        edit_link = "main",
    ),
    pages = [
        "Home" => "index.md",
        "Installation" => "installation.md",
        "Quick Start" => "quickstart.md",
        "Tutorials" => [
            "Batch Training" => "tutorials/batch_training.md",
            "Episodic Training" => "tutorials/episodic_training.md",
            "Inference & Evaluation" => "tutorials/inference.md",
        ],
        "Manual" => [
            "Architecture" => "manual/architecture.md",
            "Bundle Methods" => "manual/bundle_methods.md",
            "Data Formats" => "manual/data_formats.md",
        ],
        "API Reference" => [
            "Training Functions" => "api/training.md",
            "Testing Functions" => "api/testing.md",
            "Utilities" => "api/utilities.md",
        ],
        "Examples" => "examples.md",
    ],
    repo = Documenter.Remotes.GitLab("demelas", "bundlenetwork.jl"),
    authors = "Francesco Demelas <your.email@example.com>",
)

# Remove deploydocs for GitLab - CI handles deployment
