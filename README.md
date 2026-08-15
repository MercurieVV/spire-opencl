# spire-opencl

[![Maven Central](https://img.shields.io/maven-central/v/io.github.mercurievv/spire-opencl_3.svg)](https://central.sonatype.com/artifact/io.github.mercurievv/spire-opencl_3)

Instantiate ordinary Spire numeric code at a symbolic value type and compile the resulting expression
tree to an OpenCL kernel.

Main documentation: https://mercurievv.github.io/spire-opencl/

## Quickstart

Add the dependency:

```scala
libraryDependencies += "io.github.mercurievv" %% "spire-opencl" % "0.1.4"
```

Mill:

```scala
def mvnDeps = Seq(mvn"io.github.mercurievv::spire-opencl:0.1.4")
```

Check that your machine has a usable OpenCL device:

```bash
./mill spireOpencl.runMain io.github.mercurievv.spireopencl.opencl.DeviceProbe
```

Build docs locally:

```bash
./mill docs.site
open out/docs/site.dest/index.html
```

Run tests:

```bash
./mill spireOpencl.test
```

## Spire Features Used

- `Field`: `+`, `-`, `*`, `/`, `zero`, `one`
- `Trig`: `sin`, `cos`, `tan`, inverse trig, hyperbolic functions, `exp`, `log`, `e`, `pi`
- `NRoot`: `sqrt`, `nroot`, `fpow`
- `ConvertableTo`: numeric literals enter the expression tree
- `OrderS`: this library's mask-based comparisons, where `<` and `>` produce `0/1` values for
  branchless expressions

## Other Features

- `Expr.Const`, `Expr.Uniform`, `Expr.Param`, and `Expr.Index` cover literal, per-launch,
  per-batch-element, and per-work-item inputs.
- `Expr.Sum` reduces across the batch dimension in the generated kernel.
- Kernels use single precision on devices without `cl_khr_fp64`; `DeviceProbe` reports what the
  current device actually supports.
- Benchmarks live in `bench/`, with generated results shown on the
  [Benchmarking](https://mercurievv.github.io/spire-opencl/benchmarking.html) page.

## Release

The git tag is the published version:

```bash
git tag v0.1.5
git push --tags
```

CI tests on pushes and publishes to Maven Central on `v*` tags.

## Licence

Apache 2.0.
