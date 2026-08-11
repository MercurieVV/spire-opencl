package io.github.mercurievv.spireopencl.opencl

import cats.effect.{Resource, Sync}

import io.github.mercurievv.spireopencl.symbolic.Formula
import org.jocl.*
import org.jocl.CL.*

/** An opaque, immutable, compiled artifact.
  *
  * Composition is over by the time one of these exists — there is no way to recombine two kernels. Everything variable arrives as an argument, so one
  * compile serves every launch.
  *
  * Not thread-safe: `clSetKernelArg` mutates the kernel object, so a `Kernel` belongs to one thread.
  */
trait Kernel[F[_]]:
  /** The OpenCL C this was built from. Kept for diagnosis — reading it is how you check what the IR actually became. */
  def source: String

  /** How many batch elements one launch can carry. */
  def maxBatch: Int

  /** Whether this kernel reduces over the batch itself. A reduced kernel writes `size` floats — the total — instead of one slice per element. */
  def reduced: Boolean

  /** Fill `out` for a single batch element. `uniforms` and `params` supply the formula's declared arguments. */
  def render(uniforms: Map[String, Float], params: Map[String, Float], out: Array[Float]): F[Unit]

  /** The same work with no effect wrapper. For measurement loops that must not charge scheduling to the kernel, and for callers already inside a
    * `delay`. Blocks until the results are in `out`.
    */
  def renderUnsafe(uniforms: Map[String, Float], params: Map[String, Float], out: Array[Float]): Unit

  /** Every element of `batch`, in **one** launch. Unless the kernel reduces, `out` must hold `batch.size * size` floats and comes back element-major:
    * element `e` occupies `out[e * size .. e * size + size)`, matching the order of `batch`.
    *
    * This is the shape callers should use. A launch per element pays the driver's per-launch latency once per element, and at small sizes that
    * latency — not the arithmetic — is the whole cost.
    *
    * `onPhase`, if given, is called with `("upload", nanoTime)` right after the parameter buffer is enqueued, `("launch", nanoTime)` right after
    * the kernel is enqueued, and `("readback", nanoTime)` once the (blocking) result read completes — so a caller can time where a launch's cost
    * actually falls without this trait knowing anything about who's calling or why. Optional and free when unused: the default does nothing.
    */
  def renderBatchUnsafe(
    uniforms: Map[String, Float],
    batch: Seq[Map[String, Float]],
    out: Array[Float],
    onPhase: (String, Long) => Unit = (_, _) => (),
  ): Unit

  /** The persistent cells as the last launch left them, element-major: element `e`'s cells occupy `out[e * states .. e * states + states)`, in declared
    * order. For tests and diagnosis — the audio path never reads state back, which is the point of keeping it on the device.
    */
  def readStateUnsafe(out: Array[Float]): Unit

object ClKernel:

  final case class Device(platform: cl_platform_id, device: cl_device_id, name: String)

  /** First GPU on the first platform, or the first device of any type if there is no GPU. */
  def defaultDevice[F[_]: Sync]: F[Device] = Sync[F].delay {
    setExceptionsEnabled(true)
    val platformCount = new Array[Int](1)
    clGetPlatformIDs(0, null, platformCount)
    if platformCount(0) == 0 then throw new IllegalStateException("no OpenCL platform available")
    val platforms = new Array[cl_platform_id](platformCount(0))
    clGetPlatformIDs(platforms.length, platforms, null)

    def devicesOf(p: cl_platform_id, kind: Long): Vector[cl_device_id] =
      val count = new Array[Int](1)
      try clGetDeviceIDs(p, kind, 0, null, count)
      catch case _: CLException => count(0) = 0
      if count(0) == 0 then Vector.empty
      else
        val ids = new Array[cl_device_id](count(0))
        clGetDeviceIDs(p, kind, ids.length, ids, null)
        ids.toVector

    val candidates = platforms.toVector.flatMap(p => devicesOf(p, CL_DEVICE_TYPE_GPU).map(p -> _)) ++
      platforms.toVector.flatMap(p => devicesOf(p, CL_DEVICE_TYPE_ALL).map(p -> _))
    val (platform, device) = candidates.headOption.getOrElse(throw new IllegalStateException("no OpenCL device available"))

    val size = new Array[Long](1)
    clGetDeviceInfo(device, CL_DEVICE_NAME, 0, null, size)
    val buffer = new Array[Byte](size(0).toInt)
    clGetDeviceInfo(device, CL_DEVICE_NAME, buffer.length.toLong, Pointer.to(buffer), null)
    Device(platform, device, new String(buffer, 0, math.max(buffer.length - 1, 0)).trim)
  }

  /** Largest work-group the device will accept. A reduced kernel makes one work-group per work-item in dimension 0, sized by the batch, so this is a
    * hard ceiling on batch size for that shape.
    */
  private def maxWorkGroupSize(device: cl_device_id): Long =
    val value = new Array[Long](1)
    clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE, Sizeof.size_t.toLong, Pointer.to(value), null)
    value(0)

  private def buildLog(program: cl_program, device: cl_device_id): String =
    val size = new Array[Long](1)
    clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, null, size)
    val buffer = new Array[Byte](size(0).toInt)
    clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, buffer.length.toLong, Pointer.to(buffer), null)
    new String(buffer, 0, math.max(buffer.length - 1, 0)).trim

  /** Batch elements a single launch carries unless the caller asks for more. */
  val DefaultMaxBatch: Int = 16

  /** Compose → compile. Expensive (source generation plus a driver compile), so it belongs off any latency-critical path — at startup, or on a
    * background fiber.
    *
    * `size` is the dimension-0 extent of a launch: how many work-items the kernel runs per batch element.
    */
  def compile[F[_]: Sync](formula: Formula, size: Int, maxBatchSize: Int = DefaultMaxBatch): Resource[F, Kernel[F]] =
    Resource.eval(defaultDevice[F]).flatMap(compileOn(_, formula, size, maxBatchSize))

  def compileOn[F[_]: Sync](dev: Device, formula: Formula, size: Int, maxBatchSize: Int = DefaultMaxBatch): Resource[F, Kernel[F]] =
    val src = CodeGen(formula)
    val elementParams = formula.params
    val uniformNames = formula.uniforms
    val stateNames = formula.states

    def acquire[A](f: => A)(release: A => Unit): Resource[F, A] =
      Resource.make(Sync[F].delay(f))(a => Sync[F].delay(release(a)))

    /* Checked here, at compile, because the alternative is discovering it mid-launch. A reduced kernel puts one
     * work-group per work-item in dimension 0 with one work-item per batch element, so the device's work-group
     * ceiling is a ceiling on batch size. */
    val validate = Sync[F].delay {
      if maxBatchSize < 1 then throw new IllegalArgumentException(s"maxBatch must be at least 1, got $maxBatchSize")
      val ceiling = maxWorkGroupSize(dev.device)
      if formula.isReduced && maxBatchSize > ceiling then
        throw new IllegalArgumentException(
          s"a reduced kernel needs a work-group of $maxBatchSize elements, but ${dev.name} allows at most $ceiling",
        )
    }

    for
      _ <- Resource.eval(validate)
      context <- acquire {
        val props = new cl_context_properties()
        props.addProperty(CL_CONTEXT_PLATFORM.toLong, dev.platform)
        clCreateContext(props, 1, Array(dev.device), null, null, null)
      }(clReleaseContext)
      // OpenCL 1.2 entry point: Apple's implementation has no clCreateCommandQueueWithProperties.
      queue <- acquire(clCreateCommandQueue(context, dev.device, 0L, null))(clReleaseCommandQueue)
      program <- acquire {
        val p = clCreateProgramWithSource(context, 1, Array(src), Array(src.length.toLong), null)
        try clBuildProgram(p, 0, null, null, null, null)
        catch case e: CLException => throw new IllegalStateException(s"kernel build failed: ${e.getMessage}\n${buildLog(p, dev.device)}", e)
        p
      }(clReleaseProgram)
      kernel <- acquire(clCreateKernel(program, CodeGen.kernelName, null))(clReleaseKernel)
      outBuffer <- acquire(clCreateBuffer(context, CL_MEM_WRITE_ONLY, (Sizeof.cl_float * size * maxBatchSize).toLong, null, null))(clReleaseMemObject)
      // One packed buffer for every per-element parameter of every element, laid out element-major to match CodeGen.
      paramBuffer <- acquire(
        clCreateBuffer(context, CL_MEM_READ_ONLY, math.max(1, Sizeof.cl_float * elementParams.size * maxBatchSize).toLong, null, null),
      )(clReleaseMemObject)
      /* Two of them, swapped after every launch, because a launch reads state in every dimension-0 work-group and
       * writes it in one: reads and the write live in different work-groups, OpenCL 1.2 has no barrier across them,
       * and writing in place would therefore be a race. The pair costs one allocation at compile and a pointer swap
       * per launch — nothing per chunk. Zeroed at acquire so a formula's first launch sees a defined value. */
      stateBuffers <- acquire {
        val floats = math.max(1, stateNames.size * maxBatchSize)
        val zeros = Pointer.to(new Array[Float](floats))
        Array.fill(2)(clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, (Sizeof.cl_float * floats).toLong, zeros, null))
      }(_.foreach(clReleaseMemObject))
    yield new Kernel[F]:
      val source: String = src
      val maxBatch: Int = maxBatchSize
      val reduced: Boolean = formula.isReduced

      /** Direct, so the parameter upload can be non-blocking: JOCL rejects a non-blocking transfer from a heap array, and a blocking one costs a full
        * device round trip — measurably more than the kernel itself at small sizes.
        */
      private val staging =
        java.nio.ByteBuffer
          .allocateDirect(math.max(1, Sizeof.cl_float * elementParams.size * maxBatchSize))
          .order(java.nio.ByteOrder.nativeOrder)
          .asFloatBuffer

      /** Which of the pair the next launch reads. The other is the one it writes. */
      private val readSide = new java.util.concurrent.atomic.AtomicInteger(0)

      def readStateUnsafe(out: Array[Float]): Unit =
        if stateNames.nonEmpty then
          val floats = math.min(out.length, stateNames.size * maxBatchSize)
          clEnqueueReadBuffer(queue, stateBuffers(readSide.get), CL_TRUE, 0, (Sizeof.cl_float * floats).toLong, Pointer.to(out), 0, null, null)
        ()

      def render(uniforms: Map[String, Float], params: Map[String, Float], out: Array[Float]): F[Unit] =
        Sync[F].delay(renderUnsafe(uniforms, params, out))

      def renderUnsafe(uniforms: Map[String, Float], params: Map[String, Float], out: Array[Float]): Unit =
        renderBatchUnsafe(uniforms, Seq(params), out)

      def renderBatchUnsafe(
        uniforms: Map[String, Float],
        batch: Seq[Map[String, Float]],
        out: Array[Float],
        onPhase: (String, Long) => Unit = (_, _) => (),
      ): Unit = {
        val n = batch.size
        val outputFloats = if reduced then size else size * n
        if n == 0 then java.util.Arrays.fill(out, 0.0f)
        else if n > maxBatchSize then throw new IllegalArgumentException(s"$n batch elements exceed the compiled maximum of $maxBatchSize")
        else if out.length < outputFloats then
          throw new IllegalArgumentException(s"output array is ${out.length}, needs at least $outputFloats for $n elements of $size")
        else
          if elementParams.nonEmpty then
            batch.zipWithIndex.foreach { case (v, elementIdx) =>
              elementParams.zipWithIndex.foreach { case (name, paramIdx) =>
                val value =
                  v.getOrElse(name, throw new IllegalArgumentException(s"batch element $elementIdx is missing parameter '$name'; got ${v.keys.toList}"))
                staging.put(elementIdx * elementParams.size + paramIdx, value)
              }
            }
            staging.position(0)
            clEnqueueWriteBuffer(
              queue,
              paramBuffer,
              CL_FALSE,
              0,
              (Sizeof.cl_float * elementParams.size * n).toLong,
              Pointer.toBuffer(staging),
              0,
              null,
              null,
            )
          onPhase("upload", System.nanoTime())
          /* Bound in the order CodeGen declares them, by a running counter rather than by arithmetic over which
           * optional arguments are present: each optional argument used to shift the index of every later one. */
          val argIdx = new java.util.concurrent.atomic.AtomicInteger(0)
          def bind(size: Long, value: Pointer): Unit = clSetKernelArg(kernel, argIdx.getAndIncrement(), size, value)

          bind(Sizeof.cl_mem.toLong, Pointer.to(outBuffer))
          uniformNames.foreach { name =>
            val value = uniforms.getOrElse(name, throw new IllegalArgumentException(s"missing uniform '$name'; got ${uniforms.keys.toList}"))
            bind(Sizeof.cl_float.toLong, Pointer.to(Array(value)))
          }
          // A reduced kernel loops over the batch itself, so the count is an argument rather than an NDRange dimension.
          if reduced then bind(Sizeof.cl_int.toLong, Pointer.to(Array(n)))
          if elementParams.nonEmpty then bind(Sizeof.cl_mem.toLong, Pointer.to(paramBuffer))
          val writeSide = 1 - readSide.get
          if stateNames.nonEmpty then
            /* Elements past this launch's batch are not written by the kernel, and the buffer it writes into holds
             * what it held two launches ago — so their cells would silently rewind. Carry them across instead. Free
             * for a caller that always sends a full batch, which is the shape that keeps a slot bound to an element
             * in the first place. */
            if n < maxBatchSize then
              val offset = (Sizeof.cl_float * stateNames.size * n).toLong
              clEnqueueCopyBuffer(
                queue,
                stateBuffers(readSide.get),
                stateBuffers(writeSide),
                offset,
                offset,
                (Sizeof.cl_float * stateNames.size * (maxBatchSize - n)).toLong,
                0,
                null,
                null,
              )
            bind(Sizeof.cl_mem.toLong, Pointer.to(stateBuffers(readSide.get)))
            bind(Sizeof.cl_mem.toLong, Pointer.to(stateBuffers(writeSide)))
          if reduced then
            // A null pointer with a size allocates work-group-local memory: one float per element for the reduction.
            bind((Sizeof.cl_float * n).toLong, null)
            // Work-group = one work-item's batch, so the reduction is local to the group and needs no global sync.
            clEnqueueNDRangeKernel(queue, kernel, 2, null, Array(size.toLong, n.toLong), Array(1L, n.toLong), 0, null, null)
          else clEnqueueNDRangeKernel(queue, kernel, 2, null, Array(size.toLong, n.toLong), null, 0, null, null)
          // The side just written becomes the side the next launch reads.
          if stateNames.nonEmpty then readSide.set(writeSide)
          onPhase("launch", System.nanoTime())
          clEnqueueReadBuffer(queue, outBuffer, CL_TRUE, 0, (Sizeof.cl_float * outputFloats).toLong, Pointer.to(out), 0, null, null)
          onPhase("readback", System.nanoTime())
        ()
      }
