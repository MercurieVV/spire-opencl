package io.github.mercurievv.spireopencl.util

import scala.compiletime.{constValue, erasedValue}
import scala.deriving.Mirror

/** Field names of a single-type-param case class, as a value shaped like the case class itself — `labeled[Point]` for
  * `case class Point[V](x: V, y: V, z: V)` is `Point[String]("x", "y", "z")`, not a bare `List("x", "y", "z")`. The labels then line up with any
  * other `F[A]` of the same shape by *position*, the same way `F[A]`'s own fields line up with each other — which is what [[fill]] and [[read]] are.
  *
  * Same idea as `minuscles`' `fields-names` module (a `FieldsNames[T]` filled with its own field names via `Magnolia`), narrowed to the one shape
  * this project needs — a case class with exactly one type parameter, substituted uniformly at every field — and derived directly off the Scala 3
  * `Mirror` instead of a `Magnolia` dependency.
  */
private[spireopencl] object FieldLabels:

  /** `F[String]` with every field set to its own name. */
  inline def labeled[F[_]](using m: Mirror.ProductOf[F[String]]): F[String] =
    m.fromProduct(Tuple.fromArray(namesOf[F].toArray))

  /** `F`'s field names, in declared order, as a flat list rather than shaped as `F[String]` — for a caller that wants the names themselves (e.g. to
    * declare them on `Formula`) rather than a value to zip against.
    */
  inline def namesOf[F[_]](using m: Mirror.ProductOf[F[String]]): List[String] =
    names[m.MirroredElemLabels]

  /** `labeled[F]`, mapped field-by-field into `F[A]` — the shape stays `F`, only the value at each field changes, from its own name to `f(name)`. */
  inline def fill[F[_], A](f: String => A)(using ms: Mirror.ProductOf[F[String]], ma: Mirror.ProductOf[F[A]]): F[A] =
    ma.fromProduct(Tuple.fromArray(fields(labeled[F]).map(f).toArray[Any]))

  /** `labeled[F]`'s names, paired with `value`'s fields in the same declared order — `read(Point(1, 2, 3))` is `List("x" -> 1, "y" -> 2, "z" -> 3)`.
    */
  inline def read[F[_], A](value: F[A])(using ms: Mirror.ProductOf[F[String]]): List[(String, A)] =
    fields(labeled[F]).zip(fields(value))

  private def fields[F[_], A](value: F[A]): List[A] =
    value.asInstanceOf[Product].productIterator.map(_.asInstanceOf[A]).toList

  private inline def names[T <: Tuple]: List[String] =
    inline erasedValue[T] match
      case _: EmptyTuple => Nil
      case _: (h *: t)   => constValue[h].toString :: names[t]
