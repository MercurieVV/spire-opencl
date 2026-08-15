# spire-opencl

[![Maven Central](https://img.shields.io/maven-central/v/io.github.mercurievv/spire-opencl_3.svg)](https://central.sonatype.com/artifact/io.github.mercurievv/spire-opencl_3)
[![CI](https://github.com/MercurieVV/spire-opencl/actions/workflows/ci.yml/badge.svg)](https://github.com/MercurieVV/spire-opencl/actions/workflows/ci.yml)
[![Docs](https://github.com/MercurieVV/spire-opencl/actions/workflows/docs.yml/badge.svg)](https://mercurievv.github.io/spire-opencl/)
[![License](https://img.shields.io/github/license/MercurieVV/spire-opencl.svg)](LICENSE)

Scala 3 GPU computing for numerical code: instantiate ordinary Typelevel Spire typeclass programs at
a symbolic value type, then compile the resulting expression tree to an OpenCL kernel.

Documentation: https://mercurievv.github.io/spire-opencl/

Source: https://github.com/MercurieVV/spire-opencl

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

- Scala 3, Cats Effect, JOCL, and OpenCL 1.2.
- `Expr.Const`, `Expr.Uniform`, `Expr.Param`, and `Expr.Index` cover literal, per-launch,
  per-batch-element, and per-work-item inputs.
- `Expr.Sum` reduces across the batch dimension in the generated kernel.
- Kernels use single precision on devices without `cl_khr_fp64`; `DeviceProbe` reports what the
  current device actually supports.
- Benchmarks live in `bench/`, with generated results shown on the
  [Benchmarking](https://mercurievv.github.io/spire-opencl/benchmarking.html) page.

Useful search terms: Scala GPU, OpenCL Scala, Spire numeric typeclasses, symbolic expression compiler,
generated OpenCL kernels.

## Release

The git tag is the published version:

```bash
git tag v0.1.5
git push --tags
```

CI tests on pushes and publishes to Maven Central on `v*` tags.

## Licence

Apache 2.0.
