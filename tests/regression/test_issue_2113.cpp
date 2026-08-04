// A regression test from pocl/pocl#2113
//
// When a kernel contains a conditional while(1){} (abort-like infinite loop)
// on a path gated by get_local_id(0) % N == 0, the loop-based WG methods can
// incorrectly make later get_local_id(0) % N evaluations return 0 for every
// work-item. Auto mode should fall back to CBS for such kernels.
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

#include <iostream>
#include <vector>

constexpr int WORKGROUP_SIZE = 36;

static const char Source[] = R"OCLC(
void check_bounds(__local int *data) {
  for (int i = 0; i < 2; i++) {
    if (data[i] < 0) {
      printf("error i=%d val=%d\n", i, data[i]);
      while (1) {
      }
    }
  }
}

kernel void modulo_bug_kernel(global int *output) {
  local int shared_data[2];
  int lid = get_local_id(0);
  int gid = get_global_id(0);

  if (lid == 0) {
    shared_data[0] = 1;
    shared_data[1] = 2;
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  if ((lid % 36) == 0) {
    check_bounds(shared_data);
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  int offset = lid % 36;
  output[gid] = offset;
}
)OCLC";

int main() try {
  std::vector<cl::Platform> Platforms;
  cl::Platform::get(&Platforms);
  cl::Platform Platform = Platforms.at(0);

  std::vector<cl::Device> Devices;
  Platform.getDevices(CL_DEVICE_TYPE_CPU, &Devices);
  cl::Device Dev = Devices.at(0);

  auto Ctx = cl::Context(Dev);
  auto CmdQ = cl::CommandQueue(Ctx, Dev);
  auto Prog = cl::Program(Ctx, Source);
  Prog.build(Dev);
  auto Kernel = cl::Kernel(Prog, "modulo_bug_kernel");

  std::vector<cl_int> Output(WORKGROUP_SIZE, -1);
  cl::Buffer Buf(Ctx, CL_MEM_WRITE_ONLY, sizeof(cl_int) * WORKGROUP_SIZE);
  Kernel.setArg(0, Buf);

  CmdQ.enqueueNDRangeKernel(Kernel, cl::NullRange, cl::NDRange(WORKGROUP_SIZE),
                            cl::NDRange(WORKGROUP_SIZE));
  CmdQ.enqueueReadBuffer(Buf, CL_TRUE, 0, sizeof(cl_int) * WORKGROUP_SIZE,
                         Output.data());

  for (int I = 0; I < WORKGROUP_SIZE; I++) {
    if (Output[I] != I) {
      std::cerr << "output[" << I << "]=" << Output[I] << ", expected " << I
                << "\n";
      return 1;
    }
  }

  std::cout << "OK" << std::endl;
  return 0;
} catch (cl::Error &Ex) {
  std::cerr << "OpenCL error: " << Ex.what() << " code=" << Ex.err()
            << std::endl;
  return 2;
} catch (std::exception &Ex) {
  std::cerr << "Exception: " << Ex.what() << std::endl;
  return 2;
}
