# Evolutionary Router

A genetic algorithm solving the **Travelling Salesman Problem** in real time, built as a live, tunable instrument. Cities sit on a chart; press play and watch the population evolve the shortest closed tour, generation by generation.

It is a single self-contained static site: no build step, no dependencies.

## Run it

Open `index.html` directly in a browser, or serve the folder locally:

```bash
python -m http.server 5180
# then visit http://localhost:5180
```

## What you can do

- **Watch evolution unfold.** The brass line is the best tour found so far; faint lines behind it are the rest of the population still searching.
- **Tune live.** Adjust population size, mutation rate and operator, selection pressure, elitism, crossover type, and speed, and see the effect immediately.
- **Compare runs.** *Pin run* freezes the current convergence curve as a dashed reference. Change a parameter, reset, and race the new run against it. Cities are seeded, so the comparison is fair.

## The algorithm

A real-coded genetic algorithm over tour permutations.

| GA concept | In this simulation |
|---|---|
| Chromosome | A tour: an ordered permutation of all cities |
| Fitness | `1 / tour length` (shorter is fitter) |
| Selection | Tournament (size `k`, the selection-pressure control) |
| Crossover | Order Crossover (OX) or Partially Mapped Crossover (PMX) |
| Mutation | Inversion (reverse a segment, an untangling 2-opt move) or Swap |
| Elitism | The best tours carry into the next generation untouched |

A full, illustrated walkthrough lives in [`explanation.html`](explanation.html).

## Controls

| Control | Effect |
|---|---|
| Cities | Size of the map. More cities means a vastly harder problem |
| Population | Candidate tours kept alive each generation |
| Mutation rate / operator | How much, and how, new structure is injected |
| Selection pressure | Tournament size; higher is greedier and converges faster |
| Crossover | OX or PMX recombination |
| Elitism | How many best tours are preserved each generation |
| Speed | Generations per second |

Keyboard: `Space` play/pause, `S` step one generation, `R` reset.

## Files

```
index.html            the instrument
style.css             design system (nautical chart at night)
genetic-algorithm.js  GA core + rendering + controls
explanation.html      how it works, with diagrams
assets/               favicon
```

## Note

A genetic algorithm finds a *very good* tour, not a provably optimal one. For a live, visual search across an astronomically large space, that trade is the point: you can watch a sensible answer emerge in seconds.

Built as an educational demonstration of evolutionary computation.
