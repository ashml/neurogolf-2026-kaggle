# Competition format

## From coloured grids to executable programs

Each task contains a handful of input/output examples. A grid cell is an
integer colour from 0 to 9. The challenge is to infer the latent rule—crop an
object, extend a pattern, recolour a component, reflect a shape, and so on—and
make it generalize to inputs that are not visible during development.

NeuroGolf represents a grid as a fixed tensor with shape `1 × 10 × 30 × 30`:

- one batch item;
- ten one-hot colour channels;
- a 30 × 30 canvas with the source grid aligned at the top left.

An ONNX graph reads this tensor and returns a tensor of the same envelope. The
winning colour at each output location is decoded back into the grid.

## What ONNX changes

ONNX is a graph format for portable inference. Nodes are operations such as
`Conv`, `Slice`, `ReduceMax`, `Where` and `Concat`; edges are tensors. There is
no ordinary Python control flow at evaluation time. Shapes must be static and
the operator set is constrained.

As a result, an intuitive Python rule must be compiled into tensor algebra.
For example, “keep cells that have a same-colour neighbour” can become a single
carefully weighted 3 × 3 convolution over ten colour channels.

## Scoring

For a functionally correct task:

```text
cost   = memory + parameters
points = max(1, 25 - ln(max(1, cost)))
```

`parameters` counts values stored in initializers and constants. `memory` is
the aggregate footprint of intermediate tensors. Multiply-accumulate counts
did not contribute to the final objective used by this project.

The logarithm has two practical consequences:

- halving a large cost is valuable even if the file itself barely shrinks;
- shaving a few values from an already tiny graph can still matter across 400
  independent tasks.

## Submission shape

A submission is a ZIP containing one canonically named ONNX file per task. A
safe build process therefore needs two levels of verification:

1. validate every changed model on its own;
2. audit the complete archive, including filenames, hashes, model count and
   unintended differences from the previous accepted bundle.

## How this differs from a typical Kaggle workflow

| Typical supervised competition | NeuroGolf 2026 |
| --- | --- |
| Train one statistical model | Synthesize 400 independent programs |
| Optimize a validation metric | Exact correctness is a hard gate |
| Parameters often dominate size | Intermediate tensors can dominate cost |
| Generalization comes from training | Generalization comes from the inferred rule |
| Submit predictions | Submit executable ONNX graphs |

The most productive mental model was not “make a neural network smaller,” but
“compile each visual rule into the cheapest valid tensor program.”
