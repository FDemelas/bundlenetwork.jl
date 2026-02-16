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
            "Batch Training" => "tutorials/batch_training.md",
            "Episodic Training" => "tutorials/episodic_training.md",
            "Inference & Evaluation" => "tutorials/inference.md",
        ],
        "Manual" => [
            "Architecture" => "manual/architecture.md",
        ],
        "API Reference" => [
            "Training Functions" => "api/training.md",
        ],
        "Examples" => "examples.md",
    ],
    authors = "Francesco Demelas",
)

# NO deploydocs() for GitLab Pages - CI handles it automatically
