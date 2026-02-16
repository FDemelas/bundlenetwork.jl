using Instances

"""
    my_read_dat_json(path) -> (cpuInstanceMCND, Float32)

Read a Multi-Commodity Network Design (MCND) problem instance and its known
optimal (or best-known) dual bound from a JSON file.

The JSON file is expected to have the following structure:
```json
{
  "labels": { "Ld": <dual bound value> },
  "instance": {
    "N":    <number of nodes>,
    "o":    [<origin node indices, 0-based>],
    "d":    [<destination node indices, 0-based>],
    "q":    [<commodity demands>],
    "tail": [<arc tail node indices, 0-based>],
    "head": [<arc head node indices, 0-based>],
    "f":    [<fixed arc opening costs>],
    "c":    [<arc capacities>],
    "r":    [[<routing costs, one row per arc, one column per commodity>]]
  }
}
```

Node indices in the JSON are 0-based and are converted to 1-based Julia indices.

# Arguments
- `path::String`: Path to the JSON instance file.

# Returns
- `inst::cpuInstanceMCND`: The parsed MCND instance in CPU-compatible format.
- `goldV::Float32`: The known dual bound (label) associated with the instance,
  used as a reference value for computing the optimality gap during evaluation.
"""
function my_read_dat_json(path)
    # Open and parse the JSON file
    f    = JSON.open(path, "r")
    data = JSON.parse(f)
    close(f)

    # Extract the known dual bound (label) for this instance
    goldV    = data["labels"]["Ld"]
    data_ins = data["instance"]

    # Number of nodes in the network graph
    n = data_ins["N"]

    # Parse commodity list: each commodity is (origin, destination, demand)
    # Indices are shifted from 0-based (JSON) to 1-based (Julia)
    commodities = Tuple{Int64, Int64, Int64}[
        (data_ins["o"][i] + 1, data_ins["d"][i] + 1, data_ins["q"][i])
        for i in 1:length(data_ins["q"])
    ]

    # Parse arc list: each arc is (tail node, head node)
    # Indices are shifted from 0-based (JSON) to 1-based (Julia)
    edges = Tuple{Int64, Int64}[
        (data_ins["tail"][i] + 1, data_ins["head"][i] + 1)
        for i in 1:length(data_ins["head"])
    ]

    # Fixed arc-opening costs (one per arc)
    fc = Float32.(data_ins["f"])

    # Arc capacities (one per arc)
    c = Float32.(data_ins["c"])

    # Routing cost matrix: hcat converts list-of-rows to a matrix, then transpose
    # to obtain shape (num_commodities × num_arcs) expected by cpuInstanceMCND
    r = Float32.(hcat(data_ins["r"]...))'

    return Instances.cpuInstanceMCND(n, edges, commodities, fc, r, c), Float32(goldV)
end


"""
    my_read_dat(path) -> cpuInstanceMCND

Read a Multi-Commodity Network Design (MCND) problem instance from a
tab-separated `.dat` text file.

The file format is:
```
<N>\\t<E>\\t<K>                          # Header: nodes, arcs, commodities
# For each arc e = 1..E:
<head_e>\\t<tail_e>\\t<fixed_cost_e>\\t<capacity_e>   # Arc data
# For each commodity k = 1..K (nested inside the arc loop):
<k>\\t<routing_cost_k_e>                              # Routing cost for (k, e)
# For each commodity k = 1..K (after all arcs):
<dest_node>\\t<dest_node>\\t<demand_or_negative_supply>  # Destination line
<orig_node>\\t<orig_node>\\t<supply_or_negative_demand>  # Origin line
```

!!! note
    The arc head and tail are stored in **reversed column order** in the file
    (column 2 = head, column 1 = tail), so they are swapped when parsing.

    For each commodity, the origin/destination assignment depends on the sign
    of the demand field in the destination line:
    - If `demand ≥ 0`: the destination line gives `(dest, dest, demand)` and
      the origin line gives `(orig, orig, supply)`.
    - If `demand < 0`: the roles are swapped (the "destination" line is actually
      the origin).

# Arguments
- `path::String`: Path to the `.dat` instance file.

# Returns
- `inst::cpuInstanceMCND`: The parsed MCND instance in CPU-compatible format.
"""
function my_read_dat(path)
    f   = open(path, "r")
    sep = "\t"

    # --- Header line: network dimensions ---
    line = split(readline(f), sep; keepempty = false)
    n = parse(Int64, line[1])   # Number of nodes
    E = parse(Int64, line[2])   # Number of arcs
    K = parse(Int64, line[3])   # Number of commodities

    # Pre-allocate data structures
    commodities = Tuple{Int64, Int64, Int64}[]   # (origin, destination, demand) per commodity
    edges       = Tuple{Int64, Int64}[]          # (tail, head) per arc
    fc          = zeros(E)                       # Fixed arc-opening costs
    c           = zeros(E)                       # Arc capacities
    r           = zeros(K, E)                    # Routing costs (K × E matrix)

    # --- Arc and routing cost data (one block per arc) ---
    for e in 1:E
        line = split(readline(f), sep; keepempty = false)

        # NOTE: columns are in (head, tail) order in the file; swap to (tail, head)
        push!(edges, (parse(Int64, line[2]), parse(Int64, line[1])))

        fc[e] = parse(Int64, line[3])   # Fixed cost for opening arc e
        c[e]  = parse(Int64, line[4])   # Capacity of arc e

        # Read the routing cost for each commodity on this arc
        for k in 1:K
            line     = split(readline(f), sep; keepempty = false)
            r[k, e]  = parse(Int64, line[2])   # Routing cost for commodity k on arc e
        end
    end

    # --- Commodity data (two lines per commodity: destination then origin) ---
    for k in 1:K
        lineD = split(readline(f), sep; keepempty = false)   # Destination-side line
        lineO = split(readline(f), sep; keepempty = false)   # Origin-side line

        if parse(Int64, lineD[3]) >= 0
            # Standard orientation: destination line has non-negative demand
            # (origin, destination, demand)
            push!(commodities, (
                parse(Int64, lineO[2]),   # Origin node
                parse(Int64, lineD[2]),   # Destination node
                parse(Int64, lineD[3])    # Demand (positive)
            ))
        else
            # Reversed orientation: the "destination" line is actually the origin
            # (the sign indicates the roles are swapped in the file)
            push!(commodities, (
                parse(Int64, lineD[2]),   # Actual origin node (from destination line)
                parse(Int64, lineO[2]),   # Actual destination node (from origin line)
                parse(Int64, lineO[3])    # Demand (taken from origin line)
            ))
        end
    end

    close(f)
    return Instances.cpuInstanceMCND(n, edges, commodities, fc, r, c)
end


"""
    my_read_ga_json(path) -> (cpuInstanceGA, Float32)

Read a Generalized Assignment Problem (GA) instance and its known optimal
(or best-known) primal bound from a JSON file.

The JSON file is expected to have the following structure:
```json
{
  "labels": { "Ld": <dual/primal bound value> },
  "instance": {
    "I": <number of agents>,
    "J": <number of jobs>,
    "c": [<capacity of each agent>],
    "w": [[<resource consumption w[i][j] for agent i, job j>]],
    "p": [[<profit p[i][j] for agent i, job j>]]
  }
}
```

!!! note
    The bound stored in the JSON is negated (`-data["labels"]["Ld"]`) before
    returning. This convention arises because the GA Lagrangian is formulated
    as a minimization problem internally, but the returned `goldV` should
    represent the maximization objective value.

# Arguments
- `path::String`: Path to the JSON instance file.

# Returns
- `inst::cpuInstanceGA`: The parsed GA instance in CPU-compatible format.
- `goldV::Float32`: The negated known bound for the instance, used as a
  reference value for computing the optimality gap during evaluation.
"""
function my_read_ga_json(path)
    # Open and parse the JSON file
    f    = JSON.open(path, "r")
    data = JSON.parse(f)
    close(f)

    # Negate the stored bound: the JSON stores a minimization bound,
    # but goldV should reflect the maximization objective
    goldV    = -data["labels"]["Ld"]
    data_ins = data["instance"]

    # Number of agents and jobs
    I = data_ins["I"]
    J = data_ins["J"]

    # Agent capacities (one per agent)
    c = Float32.(data_ins["c"])

    # Resource consumption matrix: hcat converts list-of-columns to a (num_resources × J) matrix
    w = Float32.(hcat(data_ins["w"]...))

    # Profit matrix: hcat converts list-of-columns to a (num_agents × J) matrix
    p = Float32.(hcat(data_ins["p"]...))

    return Instances.cpuInstanceGA(I, J, p, w, c), Float32(goldV)
end