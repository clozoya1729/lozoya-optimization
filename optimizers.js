// gradient-free
async function optimization_ant_colony(cost_fn, initialInputs, parameters = {}) {
    /*
    implements ACOR/ACO for continuous domains.
    reference:
    - Socha & Dorigo (2008): "Ant colony optimization for continuous domains"
    - python-aco, MATLAB ACO toolbox
    */
    return new Promise((resolve) => {
        const {
            iterations = 100,
            patience = iterations,
            costTolerance = 1e-8,
            colonySize = 40,               // number of ants per generation
            archiveSize = 20,              // number of archive (elite) solutions
            q = 0.5,                       // selection pressure (0.3 to 0.7 typical)
            bounds = {},                   // {x: {min, max}, ...}
            initialPheromoneScale = 1.0,
            seed = undefined,
            renderEvery = 5,
            optimizationData = {iterations: [], costs: []},
            update_inputs_and_scene = () => {
            },
            iteration_callback = () => {
            }
        } = parameters;
        const keys = Object.keys(initialInputs);
        const randn = (typeof random_normal === "function") ? random_normal : () => (Math.random() - 0.5) * 2;
        const get_bound = key => bounds[key] || {min: -1e3, max: 1e3};
        // initialize archive uniformly over bounds
        let solutionArchive = [];
        for (let i = 0; i < archiveSize; ++i) {
            const candidate = {};
            for (const key of keys) {
                const {min, max} = get_bound(key);
                candidate[key] = min + (max - min) * Math.random();
            }
            const cost = cost_fn({inputs: candidate, ...parameters});
            solutionArchive.push({inputs: candidate, cost});
        }
        solutionArchive[0] = {
            inputs: {...initialInputs},
            cost: cost_fn({inputs: {...initialInputs}, ...parameters})
        }
        solutionArchive.sort((a, b) => a.cost - b.cost);
        let bestInputs = {...solutionArchive[0].inputs};
        let bestCost = solutionArchive[0].cost;
        let iter = 0, noImprovement = 0;
        optimizationData.iterations = [];
        optimizationData.costs = [];
        // kernel weights (roulette-wheel)
        function get_weights() {
            const weights = [];
            let sum = 0;
            for (let i = 0; i < archiveSize; ++i) {
                const w = (1 / (q * archiveSize * Math.sqrt(2 * Math.PI)))
                    * Math.exp(-0.5 * (i / (q * archiveSize)) ** 2);
                weights.push(w);
                sum += w;
            }
            for (let i = 1; i < weights.length; ++i) weights[i] += weights[i - 1];
            for (let i = 0; i < weights.length; ++i) weights[i] /= sum;
            return weights;
        }

        function select_archive_idx(weights) {
            const r = Math.random();
            for (let i = 0; i < weights.length; ++i)
                if (r < weights[i]) return i;
            return weights.length - 1;
        }

        function kernel_sigma(key, idx) {
            // spread over archive for variable key
            const c = solutionArchive[idx].inputs[key];
            let s = 0;
            for (let j = 0; j < archiveSize; ++j)
                if (j !== idx) s += Math.abs(solutionArchive[j].inputs[key] - c);
            return (s / (archiveSize - 1)) || initialPheromoneScale;
        }

        function apply_bounds(candidate) {
            for (const key of keys) {
                const {min, max} = get_bound(key);
                candidate[key] = Math.max(min, Math.min(max, candidate[key]));
            }
            return candidate;
        }

        function sample_ant(weights) {
            const candidate = {};
            for (const key of keys) {
                const idx = select_archive_idx(weights);
                const center = solutionArchive[idx].inputs[key];
                const sigma = kernel_sigma(key, idx);
                candidate[key] = center + sigma * randn();
            }
            return apply_bounds(candidate);
        }

        async function step() {
            for (; iter < iterations; ++iter) {
                const weights = get_weights();
                const colony = [];
                for (let i = 0; i < colonySize; ++i) {
                    const inputs = sample_ant(weights);
                    const cost = cost_fn({inputs, ...parameters});
                    colony.push({inputs, cost});
                }
                // combine archive and colony, keep best archiveSize
                solutionArchive = [...solutionArchive, ...colony]
                    .sort((a, b) => a.cost - b.cost)
                    .slice(0, archiveSize);
                const elite = solutionArchive[0];
                if (elite.cost < bestCost - costTolerance) {
                    bestCost = elite.cost;
                    bestInputs = {...elite.inputs};
                    noImprovement = 0;
                } else {
                    noImprovement++;
                }
                optimizationData.iterations.push(iter);
                optimizationData.costs.push(bestCost);
                if (iter % renderEvery === 0) {
                    update_inputs_and_scene(bestInputs);
                    iteration_callback(optimizationData);
                }
                if (bestCost <= costTolerance || noImprovement > patience)
                    break;
                await new Promise(r => setTimeout(r, 0));
            }
            update_inputs_and_scene(bestInputs);
            iteration_callback(optimizationData);
            resolve({inputs: bestInputs, converged: bestCost <= costTolerance, cost: bestCost});
        }

        requestAnimationFrame(step);
    });
}

async function optimization_cma_es(cost_fn, initialInputs, parameters = {}) {
    /*
    reference:
    - Hansen, N. (2009). "Benchmarking a BI-population CMA-ES on the BBOB-2009 function testbed."
    - Hansen & Ostermeier (2001). "Completely derandomized self-adaptation in evolution strategies."
    - pycma
    */
    return new Promise((resolve) => {
        numeric.eig.maxiter = 5000;
        const renderEvery = parameters.renderEvery || 5;
        const maxGlobalIters = parameters.iterations || 100;
        const patience = parameters.patience || parameters.iterations;
        const costTolerance = parameters.costTolerance || 1e-8;
        const seed = parameters.seed || undefined;
        const keys = Object.keys(initialInputs);
        const N = keys.length;
        const sigma0 = parameters.sigma || 1e-1;
        const population0 = parameters.population || (4 + Math.floor(3 * Math.log(N)));
        const restartStrategy = parameters.restartStrategy || "none"; // "ipop", "bipop", "none"
        // for BIPOP bookkeeping
        let nRestarts = 0, nGlobalIter = 0, largeRuns = 0, smallRuns = 0, maxRestarts = 50;
        let runHistory = [];
        let resolved = false;
        const bounds = parameters.bounds || {};
        const optimizationData = parameters.optimizationData || {iterations: [], costs: []};
        const update_fn = parameters.update_inputs_and_scene || (() => {
        });
        const iteration_callback_fn = parameters.iteration_callback || (() => {
        });
        var randf = make_random_normal_seeded(parameters.seed);

        async function runCMAES(population, sigmaInit, maxIters) {
            const mu = Math.floor(population / 2);
            const weights = [];
            for (let i = 0; i < mu; ++i) weights.push(Math.log(mu + 0.5) - Math.log(i + 1));
            let wSum = weights.reduce((a, b) => a + b, 0);
            for (let i = 0; i < mu; ++i) weights[i] /= wSum;
            const mueff = 1.0 / weights.reduce((sum, w) => sum + w * w, 0);
            const cc = (4 + mueff / N) / (N + 4 + 2 * mueff / N);
            const cs = (mueff + 2) / (N + mueff + 5);
            const c1 = 2 / ((N + 1.3) * (N + 1.3) + mueff);
            const cmu = Math.min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((N + 2) * (N + 2) + mueff));
            const damps = 1 + 2 * Math.max(0, Math.sqrt((mueff - 1) / (N + 1)) - 1) + cs;
            let mean = keys.map(k => initialInputs[k]);
            let sigma = sigmaInit;
            let C = numeric.identity(N);
            let pc = numeric.rep([N], 0);
            let ps = numeric.rep([N], 0);
            let B = numeric.identity(N);
            let D = numeric.rep([N], 1);
            let invsqrtC = numeric.identity(N);
            let eigeneval = 0, chiN = Math.sqrt(N) * (1 - 1 / (4 * N) + 1 / (21 * N * N));
            let bestInputs = {...initialInputs}, bestCost = cost_fn({inputs: bestInputs, ...parameters});
            let totalIter = 0, iter = 0, noImprovement = 0, converged = false;

            function bound_candidate(candidate) {
                for (let i = 0; i < N; ++i) {
                    const k = keys[i];
                    if (bounds[k]) candidate[i] = Math.max(bounds[k].min, Math.min(bounds[k].max, candidate[i]));
                }
                return candidate;
            }

            for (; iter < maxIters && nGlobalIter < maxGlobalIters; iter++, nGlobalIter++) {
                if (iter - eigeneval > 1.0 / (c1 + cmu) / N / 10) {
                    try {
                        const eig = numeric.eig(matrix_force_symmetric(C));
                        D = eig.lambda.x.map(l => Math.sqrt(Math.max(l, 1e-12)));
                        B = eig.E.x;
                        const Dinv = numeric.diag(D.map(x => 1 / (x > 0 ? x : 1)));
                        invsqrtC = numeric.dot(numeric.dot(B, Dinv), numeric.transpose(B));
                    } catch (err) {
                        mean = keys.map(k => bestInputs[k]);
                        sigma = sigmaInit;
                        C = numeric.identity(N);
                        pc = numeric.rep([N], 0);
                        ps = numeric.rep([N], 0);
                        const eig = numeric.eig(numeric.identity(N));
                        D = eig.lambda.x.map(Math.sqrt);
                        B = eig.E.x;
                        invsqrtC = numeric.identity(N);
                    }
                    eigeneval = iter;
                }
                const arz = [], arx = [];
                for (let k = 0; k < population; ++k) {
                    const z = Array.from({length: N}, randf);
                    arz.push(z);
                    const Dz = numeric.mul(D, z);
                    const BDz = numeric.dot(B, Dz);
                    let x = numeric.add(mean, numeric.mul(BDz, sigma));
                    x = bound_candidate(x);
                    arx.push(x.map(v => Number.isFinite(v) ? v : 0));
                }
                const costs = arx.map(x => {
                    const cand = {};
                    for (let i = 0; i < N; ++i) cand[keys[i]] = x[i];
                    return cost_fn({inputs: cand, ...parameters});
                });
                const order = costs.map((v, i) => [v, i]).sort((a, b) => a[0] - b[0]).map(a => a[1]);
                if (costs[order[0]] < bestCost - costTolerance) {
                    for (let i = 0; i < N; ++i) bestInputs[keys[i]] = arx[order[0]][i];
                    bestCost = costs[order[0]];
                    noImprovement = 0;
                } else noImprovement++;
                let old_mean = mean.slice();
                mean = numeric.rep([N], 0);
                for (let i = 0; i < mu; ++i)
                    mean = numeric.add(mean, numeric.mul(arx[order[i]], weights[i]));
                const y = numeric.div(numeric.sub(mean, old_mean), sigma);
                let tmp = numeric.dot(invsqrtC, y);
                ps = numeric.add(numeric.mul(ps, 1 - cs), numeric.mul(tmp, Math.sqrt(cs * (2 - cs) * mueff)));
                const ps_norm = Math.sqrt(numeric.dot(ps, ps));
                const hsig = ps_norm / Math.sqrt(1 - Math.pow(1 - cs, 2 * (iter + 1))) / chiN < 1.4 + 2 / (N + 1) ? 1 : 0;
                pc = numeric.add(numeric.mul(pc, 1 - cc), numeric.mul(y, hsig * Math.sqrt(cc * (2 - cc) * mueff)));
                let rank1 = numeric.mul(outer_product(pc, pc), c1);
                let rankmu = numeric.rep([N, N], 0);
                for (let i = 0; i < mu; ++i) {
                    const dx = numeric.sub(arx[order[i]], old_mean);
                    rankmu = numeric.add(rankmu, numeric.mul(outer_product(dx, dx), weights[i]));
                }
                rankmu = numeric.mul(rankmu, cmu);
                C = numeric.add(numeric.mul(C, 1 - c1 - cmu), numeric.add(rank1, rankmu));
                matrix_clamp(C, -1e8, 1e8, 1e-12);
                sigma *= Math.exp((cs / damps) * (ps_norm / chiN - 1));
                sigma = Math.max(1e-12, Math.min(sigma, 1e6));
                optimizationData.iterations.push(nGlobalIter);
                optimizationData.costs.push(bestCost);
                if (iter % renderEvery === 0) {
                    update_fn(bestInputs);
                    iteration_callback_fn(optimizationData);
                }
                // IPOP/BIPOP Restart triggers:
                if (noImprovement >= patience || bestCost <= costTolerance) {
                    break;
                }
                // max sigma or nan mean (emergency restart)
                if (!isFinite(sigma) || mean.some(v => !isFinite(v))) break;
                await new Promise(r => setTimeout(r, 0));
            }
            update_fn(bestInputs);
            iteration_callback_fn(optimizationData);
            return {inputs: {...bestInputs}, cost: bestCost, mean: mean.slice()};
        }

        async function ipop_bipop_loop() {
            let population = population0, sigma = sigma0, lastBestCost = Infinity;
            while (nGlobalIter < maxGlobalIters && nRestarts < maxRestarts && !resolved) {
                let isLargeRun = restartStrategy === "ipop" || (restartStrategy === "bipop" && ((largeRuns <= smallRuns) || smallRuns === 0));
                if (restartStrategy === "ipop") population = population0 * Math.pow(2, nRestarts), sigma = sigma0;
                else if (restartStrategy === "bipop") {
                    if (isLargeRun) {
                        population = population0 * Math.pow(2, largeRuns);
                        sigma = sigma0;
                    } else { // small run: random sigma, small pop
                        population = Math.round(population0 * Math.pow(2, Math.random() * 2 - 1)); // varies in [pop0/2, pop0*2]
                        sigma = Math.pow(10, -2 + 2 * Math.random()); // random in [1e-2, 1]
                    }
                }
                const {inputs, cost, mean} = await runCMAES(population, sigma, patience);
                runHistory.push({inputs, cost, population, sigma});
                if (cost < lastBestCost - costTolerance) lastBestCost = cost;
                if (cost <= costTolerance || nGlobalIter >= maxGlobalIters) {
                    resolved = true;
                    resolve({inputs, converged: true, cost});
                    return;
                }
                nRestarts++;
                if (restartStrategy === "bipop") isLargeRun ? largeRuns++ : smallRuns++;
            }
            // out of restarts/iterations
            const bestRun = runHistory.reduce((a, b) => a.cost < b.cost ? a : b);
            resolve({inputs: bestRun.inputs, converged: false, cost: bestRun.cost});
        }

        if (restartStrategy === "none" || restartStrategy === undefined) {
            // standard single-run CMA-ES
            runCMAES(population0, sigma0, maxGlobalIters).then(res => resolve({...res, converged: true}));
        } else {
            ipop_bipop_loop();
        }
        requestAnimationFrame(() => {
        });
    });
}

async function optimization_genetic_algorithm(cost_fn, initialInputs, parameters = {}) {
    return new Promise((resolve) => {
        const {
            iterations = 100,
            populationSize = 40,
            mutationRate = 0.1,
            mutationScale = 0.1,
            crossoverType = "uniform", // "uniform", "one-point", "two-point", "blend"
            selectionType = "tournament", // "tournament", "roulette"
            tournamentSize = 3,
            elitismCount = 2,
            duplicateElimination = true,
            bounds = {},
            seed = undefined,
            costTolerance = 1e-6,
            patience = iterations,
            renderEvery = 5,
            optimizationData = {iterations: [], costs: []},
            update_inputs_and_scene = () => {
            },
            iteration_callback = () => {
            }
        } = parameters;
        var randf = make_random_normal_seeded(parameters.seed);
        const keys = Object.keys(initialInputs);

        function sample_individual() {
            // sample from bounds or near initial value
            const ind = {};
            for (const key of keys) {
                if (bounds[key]) {
                    ind[key] = bounds[key].min + Math.random() * (bounds[key].max - bounds[key].min);
                } else {
                    ind[key] = initialInputs[key] + randf() * 1;
                }
            }
            return ind;
        }

        function apply_bounds(ind) {
            for (const key of keys) {
                if (bounds[key]) {
                    ind[key] = Math.max(bounds[key].min, Math.min(bounds[key].max, ind[key]));
                }
            }
            return ind;
        }

        function clone(ind) {
            return {...ind};
        }

        function mutate(ind) {
            const child = clone(ind);
            for (const key of keys) {
                if (Math.random() < mutationRate) {
                    let val = child[key] + randf() * mutationScale;
                    if (bounds[key]) val = Math.max(bounds[key].min, Math.min(bounds[key].max, val));
                    child[key] = val;
                }
            }
            return child;
        }

        function crossover(a, b) {
            const child = {};
            switch (crossoverType) {
                case "uniform":
                    for (const key of keys)
                        child[key] = Math.random() < 0.5 ? a[key] : b[key];
                    break;
                case "one-point": {
                    const point = Math.floor(Math.random() * keys.length);
                    keys.forEach((k, i) => child[k] = (i < point) ? a[k] : b[k]);
                    break;
                }
                case "two-point": {
                    let [p1, p2] = [Math.floor(Math.random() * keys.length), Math.floor(Math.random() * keys.length)];
                    if (p1 > p2) [p1, p2] = [p2, p1];
                    keys.forEach((k, i) => child[k] = (i < p1 || i >= p2) ? a[k] : b[k]);
                    break;
                }
                case "blend": // BLX-alpha (alpha=0.5)
                    for (const key of keys) {
                        const min = Math.min(a[key], b[key]), max = Math.max(a[key], b[key]);
                        const alpha = 0.5;
                        const range = max - min;
                        let val = min - alpha * range + Math.random() * (range * (1 + 2 * alpha));
                        if (bounds[key]) val = Math.max(bounds[key].min, Math.min(bounds[key].max, val));
                        child[key] = val;
                    }
                    break;
                default:
                    for (const key of keys)
                        child[key] = Math.random() < 0.5 ? a[key] : b[key];
            }
            return child;
        }

        function tournamentSelect(pop, costs, k = tournamentSize) {
            let best = null, bestCost = Infinity;
            for (let i = 0; i < k; i++) {
                const idx = Math.floor(Math.random() * pop.length);
                if (costs[idx] < bestCost) {
                    best = pop[idx];
                    bestCost = costs[idx];
                }
            }
            return best;
        }

        function rouletteSelect(pop, costs) {
            // invert costs (minimization); avoid negatives
            const maxCost = Math.max(...costs);
            const fitness = costs.map(c => maxCost - c + 1e-6);
            const sumFit = fitness.reduce((a, b) => a + b, 0);
            let pick = Math.random() * sumFit, sum = 0;
            for (let i = 0; i < pop.length; i++) {
                sum += fitness[i];
                if (sum > pick) return pop[i];
            }
            return pop[pop.length - 1];
        }

        function select(pop, costs) {
            if (selectionType === "tournament")
                return tournamentSelect(pop, costs);
            return rouletteSelect(pop, costs);
        }

        // population initialization
        let population = Array.from({length: populationSize}, () => sample_individual());
        let bestInd = clone(population[0]);
        let bestCost = Infinity;
        let bestIter = 0;
        let noImprovement = 0, iter = 0;
        optimizationData.iterations = [];
        optimizationData.costs = [];

        async function step() {
            for (; iter < iterations; iter++) {
                const costs = population.map(ind => cost_fn({inputs: ind, ...parameters}));
                // sort by cost (ascending)
                const combined = population.map((ind, i) => ({ind, cost: costs[i]}));
                combined.sort((a, b) => a.cost - b.cost);
                if (combined[0].cost < bestCost - costTolerance) {
                    bestCost = combined[0].cost;
                    bestInd = clone(combined[0].ind);
                    bestIter = iter;
                    noImprovement = 0;
                } else {
                    noImprovement++;
                }
                // elitism: preserve top elites
                const newPopulation = [];
                for (let i = 0; i < elitismCount; ++i)
                    newPopulation.push(clone(combined[i].ind));
                // generate new population
                while (newPopulation.length < populationSize) {
                    const parent1 = select(population, costs);
                    const parent2 = select(population, costs);
                    let child = crossover(parent1, parent2);
                    child = mutate(child);
                    child = apply_bounds(child);
                    newPopulation.push(child);
                }
                // duplicate handling
                if (duplicateElimination) {
                    // use JSON string as hash, avoid clones except elites
                    const unique = [];
                    const seen = new Set();
                    for (const ind of newPopulation) {
                        const h = JSON.stringify(ind);
                        if (!seen.has(h)) {
                            unique.push(ind);
                            seen.add(h);
                        } else if (unique.length < populationSize) {
                            // replace duplicate with fresh random
                            unique.push(sample_individual());
                        }
                    }
                    while (unique.length < populationSize) unique.push(sample_individual());
                    population = unique;
                } else {
                    population = newPopulation;
                }
                optimizationData.iterations.push(iter);
                optimizationData.costs.push(bestCost);
                if (iter % renderEvery === 0) {
                    update_inputs_and_scene(bestInd);
                    iteration_callback(optimizationData);
                }
                if (bestCost <= costTolerance || noImprovement >= patience) break;
                await new Promise(r => setTimeout(r, 0));
            }
            update_inputs_and_scene(bestInd);
            iteration_callback(optimizationData);
            resolve({inputs: bestInd, converged: bestCost <= costTolerance, cost: bestCost, iteration: bestIter});
        }

        requestAnimationFrame(step);
    });
}

async function optimization_particle_swarm(cost_fn, initialInputs, parameters = {}) {
    /*
    reference:
    - Clerc, M., & Kennedy, J. (2002). The particle swarm—explosion, stability, and convergence in a multidimensional complex space. IEEE Transactions on Evolutionary Computation
    - Standard PSO with constriction: φ=4.1, χ≈0.7298, w=1.0, φ₁=φ₂=2.05
    - See Kennedy/Clerc constriction for canonical values
     */
    return new Promise((resolve) => {
        const {
            renderEvery = 5,
            iterations = 100,
            costTolerance = 1e-8,
            patience = iterations,
            seed = undefined,
            swarmSize = 100,
            inertia = 0.7298,
            inertiaStart = 0.9,
            inertiaEnd = 0.4,
            velocityUpdate = "Fixed Inertia",
            cognitive = 1.49618,
            social = 1.49618,
            velocityClamp = null,
            boundaryStrategy = "reflect",
            topology = "global",
            neighborhoodSize = 3, // for ring and random
            bounds = {},
            optimizationData = {iterations: [], costs: []},
            update_inputs_and_scene = () => {
            },
            iteration_callback = () => {
            }
        } = parameters;
        var randf = make_random_normal_seeded(parameters.seed);
        const keys = Object.keys(initialInputs);

        function apply_bounds(pos) {
            for (const k of keys) {
                if (!Number.isFinite(pos[k])) pos[k] = bounds[k]?.min ?? 0;
                if (bounds[k]) {
                    if (boundaryStrategy === "clamp" || !boundaryStrategy) {
                        pos[k] = Math.max(bounds[k].min, Math.min(bounds[k].max, pos[k]));
                    } else if (boundaryStrategy === "reflect") {
                        if (pos[k] < bounds[k].min)
                            pos[k] = bounds[k].min + (bounds[k].min - pos[k]);
                        if (pos[k] > bounds[k].max)
                            pos[k] = bounds[k].max - (pos[k] - bounds[k].max);
                        pos[k] = Math.max(bounds[k].min, Math.min(bounds[k].max, pos[k]));
                    } else if (boundaryStrategy === "wrap") {
                        const range = bounds[k].max - bounds[k].min;
                        if (range > 0) {
                            while (pos[k] < bounds[k].min) pos[k] += range;
                            while (pos[k] > bounds[k].max) pos[k] -= range;
                        } else {
                            pos[k] = bounds[k].min;
                        }
                    } else if (boundaryStrategy === "random") {
                        if (pos[k] < bounds[k].min || pos[k] > bounds[k].max || !Number.isFinite(pos[k]))
                            pos[k] = bounds[k].min + Math.random() * (bounds[k].max - bounds[k].min);
                    }
                }
                if (!Number.isFinite(pos[k])) pos[k] = bounds[k]?.min ?? 0;
            }
            return pos;
        }

        function random_position() {
            const pos = {};
            for (const k of keys) {
                if (bounds[k]) pos[k] = bounds[k].min + Math.random() * (bounds[k].max - bounds[k].min);
                else pos[k] = initialInputs[k] + randf();
            }
            return pos;
        }

        function random_velocity() {
            const vel = {};
            for (const k of keys) {
                let v;
                if (velocityClamp && velocityClamp.min !== undefined && velocityClamp.max !== undefined)
                    v = random_float(velocityClamp.min, velocityClamp.max);
                else v = random_float(-Math.abs((bounds[k]?.max || 1) - (bounds[k]?.min || 0)), Math.abs((bounds[k]?.max || 1) - (bounds[k]?.min || 0)));
                vel[k] = v;
            }
            return vel;
        }

        const swarm = [];
        let globalBest = {...initialInputs}, globalBestCost = cost_fn({inputs: globalBest, ...parameters});
        let noImprovement = 0, bestIter = 0, totalIter = 0, converged = false;
        for (let i = 0; i < swarmSize; ++i) {
            const pos = random_position();
            const vel = random_velocity();
            const cost = cost_fn({inputs: pos, ...parameters});
            swarm.push({
                position: {...pos},
                velocity: {...vel},
                bestPosition: {...pos},
                bestCost: cost,
                index: i
            });
            if (cost < globalBestCost) {
                globalBestCost = cost;
                globalBest = {...pos};
                bestIter = 0;
            }
        }
        optimizationData.iterations = [];
        optimizationData.costs = [];
        let iter = 0;

        // neighborhood helper
        function get_neighborhood_indices(idx) {
            if (topology === "global") return Array.from({length: swarmSize}, (_, i) => i);
            if (topology === "ring") {
                const half = Math.floor(neighborhoodSize / 2);
                const result = [];
                for (let offset = -half; offset <= half; ++offset) {
                    let nidx = (idx + offset + swarmSize) % swarmSize;
                    result.push(nidx);
                }
                return result;
            }
            if (topology === "random") {
                const pool = Array.from({length: swarmSize}, (_, i) => i);
                for (let i = pool.length - 1; i > 0; i--) {
                    const j = Math.floor(Math.random() * (i + 1));
                    [pool[i], pool[j]] = [pool[j], pool[i]];
                }
                return pool.slice(0, neighborhoodSize);
            }
            return Array.from({length: swarmSize}, (_, i) => i); // fallback: global
        }

        function get_neighborhood_best(idx) {
            let bestCost = Infinity, bestPos = null;
            const indices = get_neighborhood_indices(idx);
            for (const n of indices) {
                if (swarm[n].bestCost < bestCost) {
                    bestCost = swarm[n].bestCost;
                    bestPos = swarm[n].bestPosition;
                }
            }
            return bestPos;
        }

        async function step() {
            for (; iter < iterations; ++iter) {
                let w = inertia;
                let chi = 1.0;
                if (velocityUpdate === "Adaptive Inertia") {
                    w = inertiaStart + (inertiaEnd - inertiaStart) * (iter / iterations);
                } else if (velocityUpdate === "Constriction") {
                    const phi = cognitive + social;
                    chi = 2 / Math.abs(2 - phi - Math.sqrt(phi * phi - 4 * phi));
                    w = 1.0;
                }
                for (const p of swarm) {
                    const neighborBest = (topology === "global") ? globalBest : get_neighborhood_best(p.index);
                    for (const k of keys) {
                        const rp = Math.random(), rg = Math.random();
                        let v = w * p.velocity[k]
                            + cognitive * rp * (p.bestPosition[k] - p.position[k])
                            + social * rg * (neighborBest[k] - p.position[k]);
                        v *= chi;
                        if (velocityClamp && typeof velocityClamp.min === 'number' && typeof velocityClamp.max === 'number') {
                            v = Math.max(velocityClamp.min, Math.min(velocityClamp.max, v));
                        }
                        p.velocity[k] = v;
                        p.position[k] += p.velocity[k];
                    }
                    apply_bounds(p.position);
                    const cost = cost_fn({inputs: p.position, ...parameters});
                    if (cost < p.bestCost) {
                        p.bestCost = cost;
                        p.bestPosition = {...p.position};
                        if (cost < globalBestCost - costTolerance) {
                            globalBestCost = cost;
                            globalBest = {...p.position};
                            bestIter = iter;
                            noImprovement = 0;
                        }
                    }
                }
                optimizationData.iterations.push(totalIter++);
                optimizationData.costs.push(globalBestCost);
                if (iter % renderEvery === 0) {
                    update_inputs_and_scene(globalBest);
                    iteration_callback(optimizationData);
                }
                if (globalBestCost <= costTolerance) {
                    converged = true;
                    break;
                }
                if (++noImprovement >= patience) break;
                await new Promise(r => setTimeout(r, 0));
            }
            update_inputs_and_scene(globalBest);
            iteration_callback(optimizationData);
            resolve({inputs: globalBest, converged, cost: globalBestCost});
        }

        requestAnimationFrame(step);
    });
}

async function optimization_simulated_annealing(cost_fn, initialInputs, parameters = {}) {
    return new Promise((resolve) => {
        const {
            renderEvery = 5,
            iterations = 100,
            patience = 500,
            costTolerance = 1e-5,
            seed = undefined,
            initialTemperature = 100,
            minTemperature = 1e-6,
            coolingRate = 0.95,
            alpha = 0.1, // for adaptive
            coolingSchedule = "geometric", // "geometric", "logarithmic", "adaptive"
            perturbationScale = 0.1,
            perturbationType = "gaussian", // "gaussian", "cauchy", "uniform", "adaptive"
            kernelSuccessWindow = 30,
            kernelSuccessRatioTarget = 0.44,
            reannealInterval = 300,
            reannealFactor = 1.5,
            hybridLocalSearch = false,
            localSearchType = "nelder-mead",
            localSearchInterval = 200,
            update_inputs_and_scene = () => {
            },
            iteration_callback = () => {
            },
            optimizationData = {iterations: [], costs: []}
        } = parameters;
        var randf = make_random_normal_seeded(parameters.seed);
        const keys = Object.keys(initialInputs);
        let currentInputs = {...initialInputs};
        let bestInputs = {...initialInputs};
        let currentCost = cost_fn({inputs: currentInputs, ...parameters});
        let bestCost = currentCost;
        let T = initialTemperature;
        let totalIter = 0, noImprovement = 0, stagnant = 0;
        let lastBestCost = bestCost, lastBestIter = 0, lastT = T;
        optimizationData.iterations = [];
        optimizationData.costs = [];
        let kernelScale = perturbationScale;
        let acceptWindow = [], acceptSuccess = [];

        function perturb(inputs, scale = 1.0, type = perturbationType) {
            const result = {};
            for (const key of keys) {
                if (type === "gaussian") result[key] = inputs[key] + scale * randf();
                else if (type === "cauchy") result[key] = inputs[key] + scale * Math.tan(Math.PI * (Math.random() - 0.5));
                else if (type === "uniform") result[key] = inputs[key] + scale * (2 * Math.random() - 1);
                else if (type === "adaptive") {
                    let s = scale;
                    if (acceptWindow.length === kernelSuccessWindow) {
                        let ratio = acceptSuccess.reduce((a, b) => a + b, 0) / kernelSuccessWindow;
                        if (ratio > kernelSuccessRatioTarget) s *= 1.2; else s *= 0.8;
                        kernelScale = s;
                        acceptWindow = [];
                        acceptSuccess = [];
                    }
                    result[key] = inputs[key] + s * randf();
                }
            }
            return result;
        }

        async function local_search(startInputs) {
            let method = localSearchType;
            let x = {...startInputs}, fval = cost_fn({inputs: x, ...parameters});
            const step = 1e-2;
            for (let iter = 0; iter < 100; iter++) {
                let grad = {};
                for (const key of keys) {
                    const orig = x[key];
                    x[key] = orig + step;
                    const plus = cost_fn({inputs: x, ...parameters});
                    x[key] = orig - step;
                    const minus = cost_fn({inputs: x, ...parameters});
                    x[key] = orig;
                    grad[key] = (plus - minus) / (2 * step);
                }
                let gnorm = Math.sqrt(keys.reduce((s, k) => s + grad[k] * grad[k], 0));
                if (gnorm < 1e-6) break;
                for (const key of keys) x[key] -= step * grad[key];
                let newFval = cost_fn({inputs: x, ...parameters});
                if (newFval < fval) fval = newFval; else break;
            }
            return {inputs: x, cost: fval};
        }

        async function step() {
            for (let iter = 0; iter < iterations; iter++) {
                let scale = (perturbationType === "adaptive") ? kernelScale : perturbationScale;
                const candidate = perturb(currentInputs, scale, perturbationType);
                const candidateCost = cost_fn({inputs: candidate, ...parameters});
                const deltaCost = candidateCost - currentCost;
                let accept = deltaCost < 0 || Math.random() < Math.exp(-deltaCost / T);
                if (perturbationType === "adaptive") {
                    acceptWindow.push(1);
                    acceptSuccess.push(accept ? 1 : 0);
                    if (acceptWindow.length > kernelSuccessWindow) {
                        acceptWindow.shift();
                        acceptSuccess.shift();
                    }
                }
                if (accept) {
                    currentInputs = candidate;
                    currentCost = candidateCost;
                    if (currentCost < bestCost - costTolerance) {
                        bestCost = currentCost;
                        bestInputs = {...currentInputs};
                        lastBestCost = bestCost;
                        lastBestIter = iter;
                        noImprovement = 0;
                    }
                } else {
                    noImprovement++;
                }
                if (coolingSchedule === "logarithmic") T = initialTemperature / Math.log(iter + 2);
                else if (coolingSchedule === "adaptive") T = Math.max(minTemperature, T * (1 - alpha));
                else T *= coolingRate;
                if (T < minTemperature) T = minTemperature;
                optimizationData.iterations.push(totalIter++);
                optimizationData.costs.push(currentCost);
                if (hybridLocalSearch && localSearchInterval > 0 && iter > 0 && iter % localSearchInterval === 0) {
                    const result = await local_search(bestInputs);
                    if (result.cost < bestCost) {
                        bestInputs = {...result.inputs};
                        bestCost = result.cost;
                        currentInputs = {...result.inputs};
                        currentCost = result.cost;
                    }
                }
                if (reannealInterval > 0 && iter > 0 && iter % reannealInterval === 0 && bestCost === lastBestCost) {
                    T = Math.max(initialTemperature * reannealFactor, T * reannealFactor);
                    lastBestCost = bestCost;
                    lastBestIter = iter;
                }
                if (noImprovement > patience || bestCost <= costTolerance || T <= minTemperature) {
                    break;
                }
                if (iter % renderEvery === 0) {
                    update_inputs_and_scene(currentInputs);
                    iteration_callback(optimizationData);
                }
                await new Promise(r => setTimeout(r, 0));
            }
            update_inputs_and_scene(bestInputs);
            iteration_callback(optimizationData);
            resolve({inputs: bestInputs, converged: true, cost: bestCost});
        }

        requestAnimationFrame(step);
    });
}

// gradient-based
async function optimization_gradient_descent(cost_fn, initialInputs, parameters = {}) {
    return new Promise((resolve) => {
        const {
            renderEvery = 5,
            iterations = 100,
            patience = 100,
            costTolerance = 1e-3,
            seed = undefined,
            optimizerType = 'adam',
            alphaInit = 0.1,
            beta1 = 0.9,
            beta2 = 0.999,
            epsilon = 1e-8,
            rho = 0.95,
            momentum = 0.9,
            historySize = 10,
            targetCostThreshold = 1e-2,
            randomRestart = true,
            restartSigma = 5,
            restartPatience = 50,
            gradClipValue = 10,
            delta = 1e-3,
            batchSize = 5,
            optimizationData = {iterations: [], costs: []},
            update_inputs_and_scene = () => {
            },
            iteration_callback = () => {
            }
        } = parameters;
        var randf = make_random_normal_seeded(parameters.seed);
        const keys = Object.keys(initialInputs);
        let inputs = {...initialInputs};
        let bestInputs = {...initialInputs};
        let bestCost = safe_cost(cost_fn({inputs, ...parameters}));
        optimizationData.iterations = [];
        optimizationData.costs = [];
        const m = {}, v = {}, velocity = {}, accum = {}, history = [], gradHist = [], gradSquares = {};
        for (const key of keys) {
            m[key] = 0;
            v[key] = 0;
            velocity[key] = 0;
            accum[key] = 0;
            gradSquares[key] = 0;
        }
        let alpha = alphaInit, iter = 0, totalIter = 0, noImprovement = 0, noRestart = 0;

        function safe_cost(val) {
            return (Number.isFinite(val) && !isNaN(val)) ? val : 1e20;
        }

        function compute_gradient(inputs) {
            const grad = {};
            for (const key of keys) {
                const original = inputs[key];
                inputs[key] = original + delta;
                const plus = safe_cost(cost_fn({inputs, ...parameters}));
                inputs[key] = original - delta;
                const minus = safe_cost(cost_fn({inputs, ...parameters}));
                inputs[key] = original;
                let g = (plus - minus) / (2 * delta);
                grad[key] = Math.max(-gradClipValue, Math.min(gradClipValue, g));
            }
            return grad;
        }

        async function step() {
            for (; iter < iterations; iter++) {
                for (let b = 0; b < batchSize; b++) {
                    const grad = compute_gradient(inputs);
                    switch (optimizerType) {
                        case 'adam':
                            for (const key of keys) {
                                m[key] = beta1 * m[key] + (1 - beta1) * grad[key];
                                v[key] = beta2 * v[key] + (1 - beta2) * grad[key] * grad[key];
                                const mHat = m[key] / (1 - Math.pow(beta1, iter + 1));
                                const vHat = v[key] / (1 - Math.pow(beta2, iter + 1));
                                const denom = Math.sqrt(vHat) + epsilon;
                                inputs[key] -= alpha * mHat / denom;
                            }
                            break;
                        case 'amsgrad':
                            for (const key of keys) {
                                m[key] = beta1 * m[key] + (1 - beta1) * grad[key];
                                v[key] = beta2 * v[key] + (1 - beta2) * grad[key] * grad[key];
                                const mHat = m[key] / (1 - Math.pow(beta1, iter + 1));
                                const vHat = v[key] / (1 - Math.pow(beta2, iter + 1));
                                gradSquares[key] = Math.max(gradSquares[key], vHat); // AMSGrad-specific v̂max
                                inputs[key] -= alpha * mHat / (Math.sqrt(gradSquares[key]) + epsilon);
                            }
                            break;
                        case 'adamw':
                            for (const key of keys) {
                                m[key] = beta1 * m[key] + (1 - beta1) * grad[key];
                                v[key] = beta2 * v[key] + (1 - beta2) * grad[key] * grad[key];
                                const mHat = m[key] / (1 - Math.pow(beta1, iter + 1));
                                const vHat = v[key] / (1 - Math.pow(beta2, iter + 1));
                                const decay = (parameters.weightDecay || 0) * inputs[key];
                                inputs[key] -= alpha * (mHat / (Math.sqrt(vHat) + epsilon) + decay);
                            }
                            break;
                        case 'nadam':
                            for (const key of keys) {
                                m[key] = beta1 * m[key] + (1 - beta1) * grad[key];
                                v[key] = beta2 * v[key] + (1 - beta2) * grad[key] * grad[key];
                                const mHat = m[key] / (1 - Math.pow(beta1, iter + 1));
                                const vHat = v[key] / (1 - Math.pow(beta2, iter + 1));
                                const nesterov = beta1 * mHat + (1 - beta1) * grad[key] / (1 - Math.pow(beta1, iter + 1));
                                inputs[key] -= alpha * nesterov / (Math.sqrt(vHat) + epsilon);
                            }
                            break;
                        case 'adamax':
                            for (const key of keys) {
                                m[key] = beta1 * m[key] + (1 - beta1) * grad[key];
                                v[key] = beta2 * v[key] + (1 - beta2) * grad[key] * grad[key];
                                const mHat = m[key] / (1 - Math.pow(beta1, iter + 1));
                                const vHat = v[key] / (1 - Math.pow(beta2, iter + 1));
                                const denom = Math.sqrt(vHat) + epsilon;
                                if (optimizerType === 'adamax') {
                                    gradSquares[key] = Math.max(beta2 * gradSquares[key], Math.abs(grad[key]));
                                    inputs[key] -= alpha * mHat / (gradSquares[key] + epsilon);
                                } else {
                                    inputs[key] -= alpha * mHat / denom;
                                }
                            }
                            break;
                        case 'rmsprop':
                            for (const key of keys) {
                                v[key] = beta2 * v[key] + (1 - beta2) * grad[key] ** 2;
                                inputs[key] -= alpha * grad[key] / (Math.sqrt(v[key]) + epsilon);
                            }
                            break;
                        case 'adagrad':
                            for (const key of keys) {
                                if (accum[key] === 0) accum[key] = 1e-8; // prevent divide by zero
                                accum[key] += grad[key] ** 2;
                                inputs[key] -= alpha * grad[key] / Math.sqrt(accum[key]);
                            }
                            break;
                        case 'adadelta':
                            for (const key of keys) {
                                v[key] = rho * v[key] + (1 - rho) * grad[key] ** 2;
                                const update = -Math.sqrt(accum[key] + epsilon) / Math.sqrt(v[key] + epsilon) * grad[key];
                                accum[key] = rho * accum[key] + (1 - rho) * update ** 2;
                                inputs[key] += update;
                            }
                            break;
                        case 'momentum':
                            for (const key of keys) {
                                velocity[key] = momentum * velocity[key] - alpha * grad[key];
                                inputs[key] += velocity[key];
                            }
                            break;
                        case 'nag':
                            for (const key of keys) {
                                // lookahead step
                                const lookahead = {...inputs};
                                lookahead[key] += momentum * velocity[key];
                                const g_look = compute_gradient(lookahead)[key];
                                // update velocity and inputs
                                velocity[key] = momentum * velocity[key] - alpha * g_look;
                                inputs[key] += velocity[key];
                            }
                            break;
                        case 'sgd':
                            for (const key of keys) {
                                const g = grad[key];
                                if (optimizerType === 'sgd') {
                                    velocity[key] = -alpha * g;
                                } else if (optimizerType === 'nag') {
                                    const lookahead = inputs[key] + momentum * velocity[key];
                                    const g_look = compute_gradient({...inputs, [key]: lookahead})[key];
                                    velocity[key] = momentum * velocity[key] - alpha * g_look;
                                } else {
                                    velocity[key] = momentum * velocity[key] - alpha * g;
                                }
                                inputs[key] += velocity[key];
                            }
                            break;
                    }
                }
                const currentCost = safe_cost(cost_fn({inputs, ...parameters}));
                if (currentCost < bestCost - costTolerance) {
                    bestCost = currentCost;
                    bestInputs = {...inputs};
                    noImprovement = 0;
                    noRestart = 0;
                } else {
                    noImprovement++;
                    noRestart++;
                }
                optimizationData.iterations.push(totalIter++);
                optimizationData.costs.push(bestCost);
                if (noImprovement >= patience) {
                    alpha = Math.max(alpha * 0.5, 1e-5);
                    noImprovement = 0;
                }
                if (randomRestart && noRestart >= restartPatience) {
                    keys.forEach(k => {
                        inputs[k] = bestInputs[k] + randf() * restartSigma;
                    });
                    noRestart = 0;
                }
                if (bestCost <= targetCostThreshold) {
                    update_inputs_and_scene(bestInputs);
                    resolve({inputs: bestInputs, converged: true, cost: bestCost});
                    return;
                }
                if (totalIter % renderEvery === 0) {
                    update_inputs_and_scene(bestInputs);
                    iteration_callback(optimizationData);
                }
                await new Promise(r => setTimeout(r, 0));
            }
            update_inputs_and_scene(bestInputs);
            resolve({inputs: bestInputs, converged: true, cost: bestCost});
        }

        requestAnimationFrame(step);
    });
}

async function optimization_second_order(cost_fn, initialInputs, parameters = {}) {
    const method = parameters.method || 'newton';
    if (method === 'newton') {
        return await optimize_newton(cost_fn, initialInputs, parameters);
    } else if (method === 'bfgs') {
        return await optimize_bfgs(cost_fn, initialInputs, parameters);
    } else if (method === 'l-bfgs') {
        return await optimize_lbfgs(cost_fn, initialInputs, parameters);
    } else {
        throw new Error(`Unknown second-order method: ${method}`);
    }
}
