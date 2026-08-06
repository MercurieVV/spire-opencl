package io.github.mercurievv.spireopencl.opencl

import org.jocl.*
import org.jocl.CL.*

/** Find out what this machine can actually run.
  *
  * Reports every platform/device, whether `cl_khr_fp64` is present (Apple Silicon GPUs do not have it, which is why the codegen emits `float`), and
  * proves the toolchain end-to-end by building and running a trivial kernel.
  *
  * Run: `mill spireOpencl.runMain io.github.mercurievv.spireopencl.opencl.DeviceProbe`
  */
object DeviceProbe:

  private def platformString(platform: cl_platform_id, param: Int): String =
    val size = new Array[Long](1)
    clGetPlatformInfo(platform, param, 0, null, size)
    val buffer = new Array[Byte](size(0).toInt)
    clGetPlatformInfo(platform, param, buffer.length.toLong, Pointer.to(buffer), null)
    new String(buffer, 0, math.max(buffer.length - 1, 0)).trim

  private def deviceString(device: cl_device_id, param: Int): String =
    val size = new Array[Long](1)
    clGetDeviceInfo(device, param, 0, null, size)
    val buffer = new Array[Byte](size(0).toInt)
    clGetDeviceInfo(device, param, buffer.length.toLong, Pointer.to(buffer), null)
    new String(buffer, 0, math.max(buffer.length - 1, 0)).trim

  private def deviceInt(device: cl_device_id, param: Int): Int =
    val value = new Array[Int](1)
    clGetDeviceInfo(device, param, Sizeof.cl_uint.toLong, Pointer.to(value), null)
    value(0)

  private def deviceLong(device: cl_device_id, param: Int): Long =
    val value = new Array[Long](1)
    clGetDeviceInfo(device, param, Sizeof.cl_ulong.toLong, Pointer.to(value), null)
    value(0)

  private def platforms(): Vector[cl_platform_id] =
    val count = new Array[Int](1)
    clGetPlatformIDs(0, null, count)
    val ids = new Array[cl_platform_id](count(0))
    clGetPlatformIDs(ids.length, ids, null)
    ids.toVector

  private def devices(platform: cl_platform_id): Vector[cl_device_id] =
    val count = new Array[Int](1)
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, null, count)
    val ids = new Array[cl_device_id](count(0))
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, ids.length, ids, null)
    ids.toVector

  private def helloSource(doublePrecision: Boolean) =
    val t = if doublePrecision then "double" else "float"
    val pragma = if doublePrecision then "#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n" else ""
    s"""${pragma}__kernel void hello(__global $t* out, $t scale) {
       |  int i = get_global_id(0);
       |  out[i] = sin(i * scale);
       |}
       |""".stripMargin

  /** Build and run [[helloSource]] on `device`, returning the samples it produced. */
  private def runHello(platform: cl_platform_id, device: cl_device_id, n: Int, doublePrecision: Boolean): Array[Double] =
    val src = helloSource(doublePrecision)
    val elemSize = if doublePrecision then Sizeof.cl_double else Sizeof.cl_float
    val props = new cl_context_properties()
    props.addProperty(CL_CONTEXT_PLATFORM.toLong, platform)
    val context = clCreateContext(props, 1, Array(device), null, null, null)
    try
      // OpenCL 1.2 (Apple) has no clCreateCommandQueueWithProperties; that is a 2.0 entry point.
      val queue = clCreateCommandQueue(context, device, 0L, null)
      try
        val program = clCreateProgramWithSource(context, 1, Array(src), Array(src.length.toLong), null)
        try
          clBuildProgram(program, 0, null, null, null, null)
          val kernel = clCreateKernel(program, "hello", null)
          try
            val bytes = (elemSize * n).toLong
            val buffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY, bytes, null, null)
            try
              clSetKernelArg(kernel, 0, Sizeof.cl_mem.toLong, Pointer.to(buffer))
              if doublePrecision then clSetKernelArg(kernel, 1, Sizeof.cl_double.toLong, Pointer.to(Array(0.01d)))
              else clSetKernelArg(kernel, 1, Sizeof.cl_float.toLong, Pointer.to(Array(0.01f)))
              clEnqueueNDRangeKernel(queue, kernel, 1, null, Array(n.toLong), null, 0, null, null)
              val out =
                if doublePrecision then
                  val d = new Array[Double](n)
                  clEnqueueReadBuffer(queue, buffer, CL_TRUE, 0, bytes, Pointer.to(d), 0, null, null)
                  d
                else
                  val f = new Array[Float](n)
                  clEnqueueReadBuffer(queue, buffer, CL_TRUE, 0, bytes, Pointer.to(f), 0, null, null)
                  f.map(_.toDouble)
              clFinish(queue)
              out
            finally clReleaseMemObject(buffer)
          finally clReleaseKernel(kernel)
        finally clReleaseProgram(program)
      finally clReleaseCommandQueue(queue)
    finally clReleaseContext(context)

  private def describe(platform: cl_platform_id, device: cl_device_id): Unit =
    val extensions = deviceString(device, CL_DEVICE_EXTENSIONS)
    println(s"  device      : ${deviceString(device, CL_DEVICE_NAME)}")
    println(s"    vendor    : ${deviceString(device, CL_DEVICE_VENDOR)}")
    println(s"    version   : ${deviceString(device, CL_DEVICE_VERSION)}")
    println(s"    units     : ${deviceInt(device, CL_DEVICE_MAX_COMPUTE_UNITS)}")
    println(s"    max group : ${deviceLong(device, CL_DEVICE_MAX_WORK_GROUP_SIZE)}")
    println(s"    fp64 ext  : ${extensions.contains("cl_khr_fp64")}  (preferred double width ${deviceInt(device, CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE)})")
    println(s"    extensions: $extensions")
    // The extension string and the preferred-width hint both lie on some drivers; the only trustworthy
    // answer is whether a double kernel actually builds and produces the right numbers.
    List(false -> "float", true -> "double").foreach { case (dp, label) =>
      val result =
        try
          val out = runHello(platform, device, 128, dp)
          val expected = math.sin(64 * 0.01)
          f"ok, out(64)=${out(64)}%.8f expected=$expected%.8f"
        catch case e: Throwable => s"FAILED: ${e.getMessage}"
      println(s"    hello $label%-6s: $result".replace("%-6s", ""))
    }

  def main(args: Array[String]): Unit =
    setExceptionsEnabled(true)
    val ps = platforms()
    println(s"OpenCL platforms: ${ps.size}")
    ps.foreach { p =>
      println(s"platform      : ${platformString(p, CL_PLATFORM_NAME)} / ${platformString(p, CL_PLATFORM_VERSION)}")
      val ds = devices(p)
      if ds.isEmpty then println("  (no devices)")
      else ds.foreach(describe(p, _))
    }
