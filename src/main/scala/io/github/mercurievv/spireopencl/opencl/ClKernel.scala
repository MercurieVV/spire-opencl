package io.github.mercurievv.spireopencl.opencl

import cats.effect.{Resource, Sync}
import io.github.mercurievv.spireopencl.symbolic.{Formula, TypedFormula}
import io.github.mercurievv.spireopencl.util.FieldLabels
import org.jocl.*
import org.jocl.CL.*

import scala.deriving.Mirror

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

  /** The dimension-0 extent this was compiled for: work-items per batch element, and the length of every input array. */
  def size: Int

  /** Whether this kernel reduces over the batch itself. A reduced kernel writes `size` floats — the total — instead of one slice per element. */
  def reduced: Boolean

  /** The formula's declared uniform names, in binding order. Exposed so a caller can split a combined case class of arguments into the uniform/param
    * maps `render`/`renderUnsafe` want — see [[ClKernel.renderUnsafeT]].
    */
  def uniformNames: List[String]

  /** The formula's declared per-element parameter names, in binding order. See [[uniformNames]]. */
  def paramNames: List[String]

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
    * `onPhase`, if given, is called with `("upload", nanoTime)` right after the parameter buffer is enqueued, `("launch", nanoTime)` right after the
    * kernel is enqueued, and `("readback", nanoTime)` once the (blocking) result read completes — so a caller can time where a launch's cost actually
    * falls without this trait knowing anything about who's calling or why. Optional and free when unused: the default does nothing.
    */
  def renderBatchUnsafe(
    uniforms: Map[String, Float],
    batch: Seq[Map[String, Float]],
    out: Array[Float],
    onPhase: (String, Long) => Unit = (_, _) => (),
  ): Unit

  /** [[renderBatchUnsafe]] with the results read into caller-owned **native** memory rather than a heap array.
    *
    * Same launch, same layout, same contract; only the readback's destination differs. It exists because a heap array is often not where the result
    * is wanted: a caller that must hand these floats to something outside the JVM — an audio host's output buffer, a native ring, a mapped file —
    * otherwise pays a copy out of the array immediately after the driver copied into it, and at small sizes that second copy is a real share of the
    * launch. Given a buffer over the destination, the driver writes it directly and there is no second copy at all.
    *
    * `out` must be **direct** — JOCL cannot take a pointer to a heap buffer, and a non-direct one is rejected here rather than deep inside the
    * driver. Results are written starting at `out.position()` and the position is left where it was, so a caller can aim successive launches at
    * successive slices of one buffer without reallocating a view.
    */
  def renderBatchIntoUnsafe(
    uniforms: Map[String, Float],
    batch: Seq[Map[String, Float]],
    out: java.nio.FloatBuffer,
    onPhase: (String, Long) => Unit = (_, _) => (),
  ): Unit

  /** [[renderBatchIntoUnsafe]] with the batch already packed, so no per-element collection is built at all.
    *
    * `params` is element-major in **declared parameter order** — element `e`'s values occupy `params[e * paramCount .. e * paramCount + paramCount)`
    * — which is the layout the staging buffer wants, so the call is a copy rather than a traversal of maps. It exists for callers on an audio path:
    * naming three floats per element through a `Map` allocated one per element per block is a real cost at a few milliseconds per block, and the
    * names are fixed at compile time anyway.
    *
    * Same direct-buffer requirement, same layout and same contract as [[renderBatchIntoUnsafe]] otherwise.
    */
  def renderBatchPackedUnsafe(
    uniforms: Map[String, Float],
    params: Array[Float],
    batchSize: Int,
    out: java.nio.FloatBuffer,
    onPhase: (String, Long) => Unit = (_, _) => (),
  ): Unit

  /** Launches, then hands `use` a **mapped** view of the output instead of copying it into caller memory.
    *
    * The ordinary readback is a device-to-host copy of every output float, and once the input arrays stay resident it is the only cost a launch still
    * has. Mapping asks the driver for a pointer to the results where they already are. On a device that shares physical memory with the host — every
    * Apple Silicon GPU, every integrated one — there is nothing to copy and the map is close to free; on a discrete device the driver still
    * transfers, but out of memory it can DMA from directly.
    *
    * Requires the kernel to have been compiled with `hostVisibleOutput = true`; the allocation flag that makes mapping cheap has to be chosen when
    * the buffer is created, and it is not free on every device, so it is the caller's decision rather than a default.
    *
    * The buffer is valid only for the duration of `use` — it is unmapped on the way out, and the device may reuse the memory for the next launch.
    * Read what is needed, or copy it, before returning. `use` is given a read-only view over exactly the floats this launch produced.
    */
  def renderBatchMappedUnsafe[A](
    uniforms: Map[String, Float],
    batch: Seq[Map[String, Float]],
    onPhase: (String, Long) => Unit = (_, _) => (),
  )(
    use: java.nio.FloatBuffer => A,
  ): A

  /** How many launches can be in flight before their results start overwriting each other. One unless the kernel was compiled for more. */
  def outputSlots: Int

  /** Enqueues a launch and returns without waiting for it, giving back the slot its results will land in.
    *
    * Every other entry point blocks on the readback, so the host pays a full round trip to the device per launch — at small sizes that round trip
    * *is* the measurement, and no amount of arithmetic changes it. Enqueuing several launches before collecting any lets the driver keep the device
    * fed and amortises the round trip across them.
    *
    * Results are only valid once [[finishUnsafe]] has returned, and only until `outputSlots` further launches have reused the slot. A caller that
    * enqueues more than the kernel has slots is silently overwriting its own results, which is why the count is fixed at compile and exposed.
    */
  def enqueueBatchUnsafe(
    uniforms: Map[String, Float],
    batch: Seq[Map[String, Float]],
    onPhase: (String, Long) => Unit = (_, _) => (),
  ): Int

  /** Blocks until everything enqueued has finished. */
  def finishUnsafe(): Unit

  /** Copies one slot's results out, for a launch already known to have finished. `out` must be direct. */
  def readSlotUnsafe(slot: Int, out: java.nio.FloatBuffer): Unit

  /** [[readSlotUnsafe]] enqueued rather than waited on, so a whole pipeline can be issued before the host stops once.
    *
    * A blocking read is a host round trip, and issuing several launches only to collect them with several blocking reads moves the round trips
    * without removing any. This is the other half: enqueue the launches, enqueue their readbacks, then wait once.
    *
    * `out` must be direct, must stay alive and untouched until [[finishUnsafe]] returns, and must be a distinct region per slot — the driver writes
    * into it whenever it gets there, and two overlapping destinations will race.
    */
  def enqueueReadSlotUnsafe(slot: Int, out: java.nio.FloatBuffer): Unit

  /** Fills a declared input array in device memory. `data` must hold at least [[size]] floats; anything beyond that is ignored.
    *
    * Separate from launching, which is the whole point: the array stays on the device and is read by every launch until it is written again. A caller
    * whose data does not change between launches uploads it once, and the per-launch transfer that would otherwise dominate disappears entirely. A
    * caller whose data does change simply writes again — no worse than sending it with the launch.
    *
    * Blocking, because it is not on the per-launch path and a caller wants to know the data is there before launching against it.
    */
  def writeInput(name: String, data: Array[Float]): F[Unit]

  /** [[writeInput]] with no effect wrapper. */
  def writeInputUnsafe(name: String, data: Array[Float]): Unit

  /** [[writeInputUnsafe]] from native memory, for data that is already outside the JVM heap. Must be **direct**; reads `size` floats from the
    * buffer's current position and leaves the position where it was.
    */
  def writeInputFromUnsafe(name: String, data: java.nio.FloatBuffer): Unit

  /** The persistent cells as the last launch left them, element-major: element `e`'s cells occupy `out[e * states .. e * states + states)`, in
    * declared order. For tests and diagnosis — the audio path never reads state back, which is the point of keeping it on the device.
    */
  def readStateUnsafe(out: Array[Float]): Unit

  /** Seeds the persistent cells the next launch will read, same element-major layout as [[readStateUnsafe]]. For tests: lets a case start a cell
    * somewhere other than zero without a launch to get it there.
    */
  def writeStateUnsafe(in: Array[Float]): Unit

object ClKernel:

  /** Where a launch's results are read back to.
    *
    * The driver only ever needs a pointer and a length, and both a heap array and a direct buffer can supply those — so the two output shapes are one
    * parameter rather than two copies of the launch body. `zeroFill` is here because an empty batch means "no work, silence" and each shape blanks
    * itself differently.
    */
  private trait Destination:
    def capacity: Int
    def zeroFill(): Unit

    /** How the finished results reach the caller, once the kernel is enqueued. Blocks until they are there.
      *
      * A method rather than a pointer for the launch body to read, because not every destination is a copy: mapping the output asks the driver for
      * the results where they already are, and there is no caller-side address to hand over.
      */
    def receive(queue: cl_command_queue, buffer: cl_mem, floats: Int): Unit

    /** Names the shape in the too-small error, which is otherwise indistinguishable between the two. */
    def describe: String

  private object Destination:

    /** The ordinary device-to-host copy, shared by the array and native-buffer shapes. */
    private def read(queue: cl_command_queue, buffer: cl_mem, floats: Int, into: Pointer): Unit =
      clEnqueueReadBuffer(queue, buffer, CL_TRUE, 0, (Sizeof.cl_float * floats).toLong, into, 0, null, null)
      ()

    def heap(out: Array[Float]): Destination = new Destination:
      def capacity: Int = out.length
      def zeroFill(): Unit = java.util.Arrays.fill(out, 0.0f)
      def describe: String = "output array"

      def receive(queue: cl_command_queue, buffer: cl_mem, floats: Int): Unit =
        read(queue, buffer, floats, Pointer.to(out))

    /** Writes from the buffer's current position and leaves the position there: the caller aims successive launches at slices of one buffer, and a
      * side effect on the position would silently make the second launch overwrite the first.
      */
    def native(out: java.nio.FloatBuffer): Destination = new Destination:
      def capacity: Int = out.remaining

      def receive(queue: cl_command_queue, buffer: cl_mem, floats: Int): Unit =
        read(queue, buffer, floats, Pointer.toBuffer(out))

      def zeroFill(): Unit =
        val base = out.position()
        val n = out.remaining
        var i = 0
        while i < n do
          out.put(base + i, 0.0f)
          i += 1

      def describe: String = "output buffer"

    /** No copy at all: the driver is asked for the results where they already are, `use` reads them there, and the mapping is released.
      *
      * `use` runs inside `receive` rather than after it because the pointer is only valid while mapped — returning it would hand the caller memory
      * the next launch is free to overwrite. Unmapping in a `finally` so a throwing `use` does not leak the mapping and wedge the queue.
      */
    def mapped[A](use: java.nio.FloatBuffer => A): Destination & (() => A) = new Destination with (() => A):
      private var result: Option[A] = None

      def capacity: Int = Int.MaxValue
      def describe: String = "mapped output"
      def apply(): A = result.getOrElse(throw new IllegalStateException("launch produced no mapped result"))

      /** An empty batch is silence: `use` still runs, over no floats, so a caller gets the same shape of answer either way. */
      def zeroFill(): Unit = result = Some(use(java.nio.FloatBuffer.allocate(0).asReadOnlyBuffer))

      def receive(queue: cl_command_queue, buffer: cl_mem, floats: Int): Unit =
        val bytes = (Sizeof.cl_float * floats).toLong
        val mapping = clEnqueueMapBuffer(queue, buffer, CL_TRUE, CL_MAP_READ, 0, bytes, 0, null, null, null)
        try result = Some(use(mapping.order(java.nio.ByteOrder.nativeOrder).asFloatBuffer.asReadOnlyBuffer))
        finally
          clEnqueueUnmapMemObject(queue, buffer, mapping, 0, null, null)
          ()

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
    clGetDeviceInfo(
      device,
      CL_DEVICE_MAX_WORK_GROUP_SIZE,
      Sizeof.size_t.toLong,
      Pointer.to(value),
      null,
    )
    value(0)

  private def buildLog(program: cl_program, device: cl_device_id): String =
    val size = new Array[Long](1)
    clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, null, size)
    val buffer = new Array[Byte](size(0).toInt)
    clGetProgramBuildInfo(
      program,
      device,
      CL_PROGRAM_BUILD_LOG,
      buffer.length.toLong,
      Pointer.to(buffer),
      null,
    )
    new String(buffer, 0, math.max(buffer.length - 1, 0)).trim

  /** Batch elements a single launch carries unless the caller asks for more. */
  val DefaultMaxBatch: Int = 16

  /** Compose → compile. Expensive (source generation plus a driver compile), so it belongs off any latency-critical path — at startup, or on a
    * background fiber.
    *
    * `size` is the dimension-0 extent of a launch: how many work-items the kernel runs per batch element.
    */
  def compile[F[_]: Sync](
    formula: Formula,
    size: Int,
    maxBatchSize: Int = DefaultMaxBatch,
    hostVisibleOutput: Boolean = false,
    outputSlots: Int = 1,
  ): Resource[F, Kernel[F]] =
    Resource.eval(defaultDevice[F]).flatMap(compileOn(_, formula, size, maxBatchSize, hostVisibleOutput, outputSlots))

  /** As `compile`, for a `TypedFormula[Args]` — `Reify.statefulVarTyped[Args]`, for instance. The `Kernel` comes back wrapped as
    * `TypedKernel[F, Args]`, whose `renderUnsafeT` accepts only `Args[Float]`: a case class built for a different formula is a compile error here,
    * not a name mismatch discovered at launch.
    */
  def compileT[F[_]: Sync, Args[_]](
    typed: TypedFormula[Args],
    size: Int,
    maxBatchSize: Int = DefaultMaxBatch,
    hostVisibleOutput: Boolean = false,
    outputSlots: Int = 1,
  ): Resource[F, TypedKernel[F, Args]] =
    compile[F](typed.formula, size, maxBatchSize, hostVisibleOutput, outputSlots).map(TypedKernel(_))

  /** `hostVisibleOutput` allocates the output where the host can map it, which is what makes [[Kernel.renderBatchMappedUnsafe]] cheap. Off by
    * default: on a device with its own memory it can place the buffer somewhere the kernel writes to more slowly, and a caller who reads back the
    * ordinary way would pay for a facility it never uses.
    */
  def compileOn[F[_]: Sync](
    dev: Device,
    formula: Formula,
    size: Int,
    maxBatchSize: Int = DefaultMaxBatch,
    hostVisibleOutput: Boolean = false,
    outputSlots: Int = 1,
  ): Resource[F, Kernel[F]] =
    val src = CodeGen(formula)
    val elementParams = formula.params
    val stateNames = formula.states
    val inputNames = formula.inputs
    // The compile parameters, captured before the returned kernel shadows their names with its own members.
    val dim0 = size
    val slots = outputSlots

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
      _       <- Resource.eval(validate)
      context <- acquire {
        val props = new cl_context_properties()
        props.addProperty(CL_CONTEXT_PLATFORM.toLong, dev.platform)
        clCreateContext(props, 1, Array(dev.device), null, null, null)
      }(clReleaseContext)
      // OpenCL 1.2 entry point: Apple's implementation has no clCreateCommandQueueWithProperties.
      queue   <- acquire(clCreateCommandQueue(context, dev.device, 0L, null))(clReleaseCommandQueue)
      program <- acquire {
        val p = clCreateProgramWithSource(context, 1, Array(src), Array(src.length.toLong), null)
        try clBuildProgram(p, 0, null, null, null, null)
        catch case e: CLException => throw new IllegalStateException(s"kernel build failed: ${e.getMessage}\n${buildLog(p, dev.device)}", e)
        p
      }(clReleaseProgram)
      kernel <- acquire(clCreateKernel(program, CodeGen.kernelName, null))(clReleaseKernel)
      /* One buffer per slot. A launch that is not read back immediately must not have its results
       * overwritten by the next one, so slots rotate: with a single slot this is exactly the old
       * behaviour, and with more the host can let several launches accumulate before collecting any. */
      outBuffers <- acquire(
        Array.fill(outputSlots)(
          clCreateBuffer(
            context,
            if hostVisibleOutput then CL_MEM_WRITE_ONLY | CL_MEM_ALLOC_HOST_PTR else CL_MEM_WRITE_ONLY,
            (Sizeof.cl_float * size * maxBatchSize).toLong,
            null,
            null,
          ),
        ),
      )(_.foreach(clReleaseMemObject))
      // One packed buffer for every per-element parameter of every element, laid out element-major to match CodeGen.
      paramBuffer <- acquire(
        clCreateBuffer(
          context,
          CL_MEM_READ_ONLY,
          math.max(1, Sizeof.cl_float * elementParams.size * maxBatchSize).toLong,
          null,
          null,
        ),
      )(clReleaseMemObject)
      /* One buffer per declared input, each the dimension-0 extent — an input is per work-item, so its length has
       * nothing to do with the batch. Zero-filled at acquire so that a formula launched before its arrays are
       * written reads defined zeros rather than whatever the driver handed back; `pendingInputs` below turns that
       * into an error instead, but the zeros mean the failure mode is deterministic either way. */
      inputBuffers <- acquire {
        val zeros = Pointer.to(new Array[Float](size))
        inputNames
          .map(_ =>
            clCreateBuffer(
              context,
              CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
              (Sizeof.cl_float * size).toLong,
              zeros,
              null,
            ),
          )
          .toArray
      }(_.foreach(clReleaseMemObject))
      /* Two of them, swapped after every launch, because a launch reads state in every dimension-0 work-group and
       * writes it in one: reads and the write live in different work-groups, OpenCL 1.2 has no barrier across them,
       * and writing in place would therefore be a race. The pair costs one allocation at compile and a pointer swap
       * per launch — nothing per chunk. Zeroed at acquire so a formula's first launch sees a defined value. */
      stateBuffers <- acquire {
        val floats = math.max(1, stateNames.size * maxBatchSize)
        val zeros = Pointer.to(new Array[Float](floats))
        Array.fill(2)(
          clCreateBuffer(
            context,
            CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
            (Sizeof.cl_float * floats).toLong,
            zeros,
            null,
          ),
        )
      }(_.foreach(clReleaseMemObject))
    yield new Kernel[F]:
      val source: String = src
      val maxBatch: Int = maxBatchSize
      val size: Int = dim0
      val outputSlots: Int = slots
      val uniformNames: List[String] = formula.uniforms
      val paramNames: List[String] = formula.params

      /** Which slot the next launch writes. Rotates, so `outputSlots` launches can be in flight before one overwrites another. */
      private val nextSlot = new java.util.concurrent.atomic.AtomicInteger(0)
      val reduced: Boolean = formula.isReduced

      /** Inputs not yet written. An unwritten array reads as zeros, which is a silent wrong answer rather than a loud one — so the first launch
        * refuses instead. One int comparison per launch, and only when the formula has inputs at all.
        */
      private var pendingInputs = inputNames.size

      private val inputWritten = new Array[Boolean](inputNames.size)

      private def inputSlot(name: String): Int =
        val idx = inputNames.indexOf(name)
        if idx < 0 then throw new IllegalArgumentException(s"undeclared input '$name'; declared: $inputNames")
        else idx

      private def markWritten(idx: Int): Unit =
        if !inputWritten(idx) then
          inputWritten(idx) = true
          pendingInputs -= 1

      def enqueueBatchUnsafe(
        uniforms: Map[String, Float],
        batch: Seq[Map[String, Float]],
        onPhase: (String, Long) => Unit = (_, _) => (),
      ): Int =
        launch(uniforms, batch.size, () => fillFromMaps(batch), None, onPhase)

      def finishUnsafe(): Unit =
        clFinish(queue)
        ()

      def readSlotUnsafe(slot: Int, out: java.nio.FloatBuffer): Unit = readSlot(slot, out, CL_TRUE)

      def enqueueReadSlotUnsafe(slot: Int, out: java.nio.FloatBuffer): Unit = readSlot(slot, out, CL_FALSE)

      private def readSlot(slot: Int, out: java.nio.FloatBuffer, blocking: Boolean): Unit =
        if slot < 0 || slot >= slots then throw new IllegalArgumentException(s"slot $slot is outside the compiled range 0..${slots - 1}")
        else if !out.isDirect then throw new IllegalArgumentException("output buffer must be direct: JOCL cannot take a pointer into the JVM heap")
        else
          val floats = math.min(out.remaining, size * maxBatchSize)
          clEnqueueReadBuffer(
            queue,
            outBuffers(slot),
            blocking,
            0,
            (Sizeof.cl_float * floats).toLong,
            Pointer.toBuffer(out),
            0,
            null,
            null,
          )
          ()

      def writeInput(name: String, data: Array[Float]): F[Unit] = Sync[F].delay(writeInputUnsafe(name, data))

      def writeInputUnsafe(name: String, data: Array[Float]): Unit =
        val idx = inputSlot(name)
        if data.length < size then throw new IllegalArgumentException(s"input '$name' is ${data.length} floats, need $size")
        else
          clEnqueueWriteBuffer(
            queue,
            inputBuffers(idx),
            CL_TRUE,
            0,
            (Sizeof.cl_float * size).toLong,
            Pointer.to(data),
            0,
            null,
            null,
          )
          markWritten(idx)

      def writeInputFromUnsafe(name: String, data: java.nio.FloatBuffer): Unit =
        val idx = inputSlot(name)
        if !data.isDirect then throw new IllegalArgumentException("input buffer must be direct: JOCL cannot take a pointer into the JVM heap")
        else if data.remaining < size then throw new IllegalArgumentException(s"input '$name' has ${data.remaining} floats remaining, need $size")
        else
          clEnqueueWriteBuffer(
            queue,
            inputBuffers(idx),
            CL_TRUE,
            0,
            (Sizeof.cl_float * size).toLong,
            Pointer.toBuffer(data),
            0,
            null,
            null,
          )
          markWritten(idx)

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
          clEnqueueReadBuffer(
            queue,
            stateBuffers(readSide.get),
            CL_TRUE,
            0,
            (Sizeof.cl_float * floats).toLong,
            Pointer.to(out),
            0,
            null,
            null,
          )
        ()

      def writeStateUnsafe(in: Array[Float]): Unit =
        if stateNames.nonEmpty then
          val floats = math.min(in.length, stateNames.size * maxBatchSize)
          clEnqueueWriteBuffer(
            queue,
            stateBuffers(readSide.get),
            CL_TRUE,
            0,
            (Sizeof.cl_float * floats).toLong,
            Pointer.to(in),
            0,
            null,
            null,
          )
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
      ): Unit =
        val _ = launch(
          uniforms,
          batch.size,
          () => fillFromMaps(batch),
          Some(Destination.heap(out)),
          onPhase,
        )

      def renderBatchPackedUnsafe(
        uniforms: Map[String, Float],
        params: Array[Float],
        batchSize: Int,
        out: java.nio.FloatBuffer,
        onPhase: (String, Long) => Unit = (_, _) => (),
      ): Unit =
        val needed = batchSize * elementParams.size
        if params.length < needed then
          throw new IllegalArgumentException(s"packed parameters are ${params.length} floats, need $needed for $batchSize elements")
        else if !out.isDirect then throw new IllegalArgumentException("output buffer must be direct: JOCL cannot take a pointer into the JVM heap")
        else
          val _ = launch(
            uniforms,
            batchSize,
            () => {
              var i = 0
              while i < needed do
                staging.put(i, params(i))
                i += 1
            },
            Some(Destination.native(out)),
            onPhase,
          )

      def renderBatchMappedUnsafe[A](
        uniforms: Map[String, Float],
        batch: Seq[Map[String, Float]],
        onPhase: (String, Long) => Unit = (_, _) => (),
      )(
        use: java.nio.FloatBuffer => A,
      ): A =
        if !hostVisibleOutput then
          throw new IllegalStateException(
            "mapping the output needs a kernel compiled with hostVisibleOutput = true; " +
              "without it the driver has nothing host-visible to map and would copy anyway",
          )
        else
          val dest = Destination.mapped(use)
          val _ = launch(uniforms, batch.size, () => fillFromMaps(batch), Some(dest), onPhase)
          dest()

      def renderBatchIntoUnsafe(
        uniforms: Map[String, Float],
        batch: Seq[Map[String, Float]],
        out: java.nio.FloatBuffer,
        onPhase: (String, Long) => Unit = (_, _) => (),
      ): Unit =
        if !out.isDirect then throw new IllegalArgumentException("output buffer must be direct: JOCL cannot take a pointer into the JVM heap")
        else
          val _ = launch(
            uniforms,
            batch.size,
            () => fillFromMaps(batch),
            Some(Destination.native(out)),
            onPhase,
          )

      /** One launch. The destination is the only thing the two public entry points disagree about, and it is reached exactly twice — once to size the
        * check, once to hand the driver a pointer — so keeping them one body is what stops the two paths drifting.
        */
      /** The staging fill for a batch of `Map`s: one lookup per declared parameter per element. */
      private def fillFromMaps(batch: Seq[Map[String, Float]]): Unit =
        batch.zipWithIndex.foreach { case (v, elementIdx) =>
          elementParams.zipWithIndex.foreach { case (name, paramIdx) =>
            val value =
              v.getOrElse(
                name,
                throw new IllegalArgumentException(s"batch element $elementIdx is missing parameter '$name'; got ${v.keys.toList}"),
              )
            staging.put(elementIdx * elementParams.size + paramIdx, value)
          }
        }

      /** `dest` is optional: `None` enqueues the work and returns without waiting for it, which is what lets a caller put several launches in flight
        * before collecting any of them. The slot the results land in is returned so they can be collected later.
        */
      private def launch(
        uniforms: Map[String, Float],
        n: Int,
        fillStaging: () => Unit,
        dest: Option[Destination],
        onPhase: (String, Long) => Unit,
      ): Int = {
        val outputFloats = if reduced then size else size * n
        val slot = if slots == 1 then 0 else math.floorMod(nextSlot.getAndIncrement(), slots)
        val outBuffer = outBuffers(slot)
        if n == 0 then dest.foreach(_.zeroFill())
        else if n > maxBatchSize then throw new IllegalArgumentException(s"$n batch elements exceed the compiled maximum of $maxBatchSize")
        else if pendingInputs > 0 then
          throw new IllegalStateException(
            s"input ${inputNames.zipWithIndex.collect { case (n, i) if !inputWritten(i) => s"'$n'" }.mkString(", ")} never written; " +
              "call writeInput before launching, or the kernel reads zeros",
          )
        else if dest.exists(_.capacity < outputFloats) then
          val d = dest.get
          throw new IllegalArgumentException(s"${d.describe} holds ${d.capacity} floats, needs at least $outputFloats for $n elements of $size")
        else
          if elementParams.nonEmpty then
            fillStaging()
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
          formula.uniforms.foreach { name =>
            val value = uniforms.getOrElse(
              name,
              throw new IllegalArgumentException(s"missing uniform '$name'; got ${uniforms.keys.toList}"),
            )
            bind(Sizeof.cl_float.toLong, Pointer.to(Array(value)))
          }
          // A reduced kernel loops over the batch itself, so the count is an argument rather than an NDRange dimension.
          if reduced then bind(Sizeof.cl_int.toLong, Pointer.to(Array(n)))
          if elementParams.nonEmpty then bind(Sizeof.cl_mem.toLong, Pointer.to(paramBuffer))
          // In declared order, matching CodeGen's argument list. Nothing is transferred here: the arrays are already resident.
          var inputIdx = 0
          while inputIdx < inputBuffers.length do
            bind(Sizeof.cl_mem.toLong, Pointer.to(inputBuffers(inputIdx)))
            inputIdx += 1
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
            clEnqueueNDRangeKernel(
              queue,
              kernel,
              2,
              null,
              Array(size.toLong, n.toLong),
              Array(1L, n.toLong),
              0,
              null,
              null,
            )
          else clEnqueueNDRangeKernel(queue, kernel, 2, null, Array(size.toLong, n.toLong), null, 0, null, null)
          // The side just written becomes the side the next launch reads.
          if stateNames.nonEmpty then readSide.set(writeSide)
          onPhase("launch", System.nanoTime())
          dest.foreach(_.receive(queue, outBuffer, outputFloats))
          onPhase("readback", System.nanoTime())
        slot
      }

/** As [[Kernel.renderUnsafe]], but the two `Map[String, Float]`s come from one case class instead — `renderUnsafeT(Args(alpha = 0.5f, x = sample),
  * out)` for `case class Args[T](alpha: T, x: T)` where `alpha` is a uniform and `x` a parameter is the same launch as
  * `renderUnsafe(Map("alpha" -> 0.5f), Map("x" -> sample), out)`.
  *
  * Which field is a uniform and which a parameter is read off the compiled kernel's own [[Kernel.uniformNames]]/[[Kernel.paramNames]], not off `In` —
  * so one case class can carry both without saying which is which twice. A field whose name matches neither is simply unused; a declared name with no
  * matching field fails fast, naming what's missing.
  *
  * Untyped: nothing here ties `In` to the formula the kernel was compiled from, so a case class meant for a different formula that happens to share a
  * field name still compiles and launches, silently reading the wrong value. Prefer [[ClKernel.compileT]] + [[TypedKernel.renderUnsafeT]], which fix
  * `In` at compile time from the formula itself — reach for this one where there is no `TypedFormula`, e.g. a kernel with array `inputs` or
  * hand-written `Formula.stateful` cells that were never given a single case-class shape.
  */
extension [F[_]](kernel: Kernel[F])

  inline def renderUnsafeT[In[_]](
    args: In[Float],
    out: Array[Float],
  )(using Mirror.ProductOf[In[String]],
    Mirror.ProductOf[In[Float]],
  ): Unit =
    val byName = FieldLabels.read[In, Float](args).toMap
    def floatField(name: String): Float =
      byName.getOrElse(
        name,
        throw new IllegalArgumentException(s"missing field '$name' in $args; declared: ${byName.keys.toList}"),
      )
    val uniforms = kernel.uniformNames.map(n => n -> floatField(n)).toMap
    val params = kernel.paramNames.map(n => n -> floatField(n)).toMap
    kernel.renderUnsafe(uniforms, params, out)

/** As [[Kernel.writeInputUnsafe]], but one call for every declared array instead of one per name — `writeInputsT(Point(xs, ys, zs))` for
  * `case class Point[V](x: V, y: V, z: V)` is `writeInputUnsafe("x", xs); writeInputUnsafe("y", ys); writeInputUnsafe("z", zs)`.
  *
  * Untyped for the same reason [[Kernel.renderUnsafeT]] is: nothing here ties `In` to the formula the kernel was compiled from. Prefer
  * [[ClKernel.compileT]] + [[TypedKernel.writeInputsT]], built from `Reify.arraysTyped[F]`, where `In` is fixed to the formula's own declared inputs.
  */
extension [F[_]](kernel: Kernel[F])

  inline def writeInputsT[In[_]](data: In[Array[Float]])(using Mirror.ProductOf[In[String]]): Unit =
    FieldLabels.read[In, Array[Float]](data).foreach { case (name, arr) => kernel.writeInputUnsafe(name, arr) }

/** A `Kernel[F]` compiled from a `TypedFormula[Args]`, still carrying `Args` — see [[ClKernel.compileT]]. Everything but the launch call stays on the
  * plain `Kernel`, reachable via `.kernel`: batching, arrays, state readback, and the rest neither know nor need `Args`.
  */
final case class TypedKernel[F[_], Args[_]](kernel: Kernel[F])

/** [[Kernel.renderUnsafeT]] pinned to the `Args` this kernel was compiled for — the whole reason `TypedKernel` exists. Passing a case class built for
  * a different `TypedFormula` is a type error here, where the untyped `Kernel.renderUnsafeT` would only catch it, if at all, as a runtime "missing
  * field" once the names happened not to match.
  */
extension [F[_], Args[_]](kernel: TypedKernel[F, Args])

  inline def renderUnsafeT(
    args: Args[Float],
    out: Array[Float],
  )(using Mirror.ProductOf[Args[String]],
    Mirror.ProductOf[Args[Float]],
  ): Unit =
    kernel.kernel.renderUnsafeT[Args](args, out)

  /** [[Kernel.writeInputsT]] pinned to the `Args` this kernel was compiled for — see [[ClKernel.compileT]] and `Reify.arraysTyped`. A `Point[Array[
    * Float]]` built for a different `TypedFormula` is a type error here rather than a name that happens, or fails, to line up.
    */
  inline def writeInputsT(data: Args[Array[Float]])(using Mirror.ProductOf[Args[String]]): Unit =
    kernel.kernel.writeInputsT[Args](data)
