// A regression test from pocl/pocl#849
//
// WorkitemLoops peeling of conditional barrier regions could miscompile
// kernels that write to local memory both inside a conditional barrier
// region and again after it, leading to heap corruption / invalid stores.
// Auto/loopvec/loops should fall back to CBS for such kernels.
//
// Copyright (c) 2026 PoCL developers
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to
// deal in the Software without restriction, including without limitation the
// rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
// sell copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
// DEALINGS IN THE SOFTWARE.

#define CL_HPP_TARGET_OPENCL_VERSION 300
#define CL_HPP_ENABLE_EXCEPTIONS
#include <CL/opencl.hpp>

#include <cassert>
#include <iostream>
#include <vector>

// Reduced reproducer from pocl/pocl#849. The post-barrier local write
// (enabled by the second if) is what triggered heap corruption under
// WorkitemLoops peeling.
static const char *Source = R"RAW(
#define lid(N) ((int) get_local_id(N))
#define gid(N) ((int) get_group_id(N))
__kernel void __attribute__ ((reqd_work_group_size(16, 16, 1))) rank_one(
    __global float *__restrict__ out, int const n)
{
  __local float fetch[16];
  if (16 + 16 * gid(0) < n)
  {
    fetch[lid(0)] = 1;
    barrier(CLK_LOCAL_MEM_FENCE);
    if (16 * gid(1) + lid(1) < n) {
      out[16 * gid(0) + lid(0) + n * (16 * gid(1) + lid(1))] = fetch[lid(0)];
    }
  }
  if (16 * gid(1) + lid(0) < n) {
    fetch[lid(0)] = 1;
  }
}
)RAW";

int main() {
  const int N = 100;
  try {
    std::vector<cl::Platform> Platforms;
    cl::Platform::get(&Platforms);
    std::vector<cl::Device> Devices;
    Platforms.at(0).getDevices(CL_DEVICE_TYPE_CPU, &Devices);
    cl::Device Dev = Devices.at(0);

    cl::Context Ctx(Dev);
    cl::CommandQueue Queue(Ctx, Dev);
    cl::Program Program(Ctx, Source);
    Program.build(Dev);

    cl::Buffer OutBuf(Ctx, CL_MEM_READ_WRITE, sizeof(float) * N * N);
    std::vector<float> Out(N * N, 0.f);
    Queue.enqueueWriteBuffer(OutBuf, CL_TRUE, 0, sizeof(float) * N * N,
                             Out.data());

    cl::Kernel Knl(Program, "rank_one");
    Knl.setArg(0, OutBuf);
    Knl.setArg(1, N);
    Queue.enqueueNDRangeKernel(
        Knl, cl::NullRange,
        cl::NDRange(16 * ((N + 15) / 16), 16 * ((N + 15) / 16)),
        cl::NDRange(16, 16));
    Queue.finish();

    Queue.enqueueReadBuffer(OutBuf, CL_TRUE, 0, sizeof(float) * N * N,
                            Out.data());

    for (int X = 0; X < N; ++X) {
      for (int Y = 0; Y < N; ++Y) {
        int Index = Y * N + X;
        int G0 = X / 16;
        if (16 + 16 * G0 < N)
          assert(Out[Index] == 1.f);
        else
          assert(Out[Index] == 0.f);
      }
    }

    std::cout << "OK\n";
    return 0;
  } catch (cl::Error &Err) {
    std::cerr << "OpenCL error: " << Err.what() << " (" << Err.err() << ")\n";
    return 2;
  }
}
