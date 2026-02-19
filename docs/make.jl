using Documenter

makedocs(
    sitename = "BundleNetworks Documentation",
    format = Documenter.HTML(
        prettyurls = false,  # Important for GitLab Pages!
        edit_link = "main",
    ),
    modules = [BundleNetworks],
    pages = [
        "Home" => "index.md",
        "Installation" => "installation.md",
        "Quick Start" => "quickstart.md",
        "Tutorials" => [
            "Batch Training" => "tutorials/batch_training.md",
            "Episodic Training" => "tutorials/episodic_training.md",
            "Inference & Evaluation" => "tutorials/testing.md",
        ],
        "Manual" => [
            "Architecture" => "manual/architecture.md",
        ],
        "API Reference" => [
            "Training Functions" => "api/training.md",
        ],
        "Examples" => "examples.md",
	    "API Reference" => [
  		  "Training Functions" => "api/training.md",
    		  "Docstrings" => "api/docstrings.md",   # ← add this
            ],
    ],
    authors = "Francesca Demelas",
)

deploydocs(
    repo = "github.com/FDemelas/bundlenetwork.jl.git",
    devbranch = "main",
)
