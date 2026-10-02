// Stream-bandwidth diagnosis matrix: where does warp's 42 GB/s cap come
// from, given torch.xpu streams 87-88 GB/s on the same device?
// Dimensions: USM type (shared vs device) x load width (scalar vs float4)
//             x memory-level parallelism (serial acc chain vs 4-way ILP).
// Read-only sum over 256 MB; the sum result is pinned to volatile to keep
// the compiler honest. Build: icx /fsycl /O2 stream_matrix.cpp /Fe:stream_matrix.exe
#include <sycl/sycl.hpp>
#include <chrono>
#include <cstdio>
#include <cstdlib>

namespace {
constexpr size_t kFloats = 64ull * 1024 * 1024;  // 256 MB
constexpr size_t kPerItem = 64;                  // floats per work-item
constexpr size_t kItems = kFloats / kPerItem;
constexpr int kIters = 10;

double now_s() {
  return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

template <int ILP>
float sum_scalar(const float* p, size_t base) {
  float acc[ILP];
  for (int c = 0; c < ILP; ++c) acc[c] = 0.0f;
  for (size_t i = 0; i < kPerItem; ++i)
    acc[i % ILP] += p[base + i];  // ILP chains; ILP=1 == warp's serial acc
  float s = 0.0f;
  for (int c = 0; c < ILP; ++c) s += acc[c];
  return s;
}

template <int VEC>
float sum_vec4(const sycl::float4* p, size_t base4) {
  // VEC float4 loads per item (VEC*4 == kPerItem), one acc per lane comp
  sycl::float4 acc(0.0f, 0.0f, 0.0f, 0.0f);
  for (int i = 0; i < VEC; ++i) acc += p[base4 + i];
  return acc.x() + acc.y() + acc.z() + acc.w();
}

template <typename T, typename Fn>
void run(sycl::queue& q, const char* name, T* in, size_t n_in, float* out,
         Fn kernel_body, size_t items, double bytes) {
  for (int w = 0; w < 3; ++w) {
    q.parallel_for(sycl::range<1>(items), [=](sycl::id<1> i) {
      out[i] = kernel_body(in, static_cast<size_t>(i) * kPerItem);
    });
  }
  q.wait();
  double t0 = now_s();
  for (int it = 0; it < kIters; ++it) {
    q.parallel_for(sycl::range<1>(items), [=](sycl::id<1> i) {
      out[i] = kernel_body(in, static_cast<size_t>(i) * kPerItem);
    });
  }
  q.wait();
  double dt = (now_s() - t0) / kIters;
  std::printf("%-42s %8.3f ms   %6.1f GB/s\n", name, dt * 1e3, bytes / 1e9 / dt);
}

template <typename T>
void run_copy(sycl::queue& q, const char* name, T* dst, const T* src, size_t bytes) {
  for (int w = 0; w < 3; ++w)
    q.memcpy(dst, src, bytes);
  q.wait();
  double t0 = now_s();
  for (int it = 0; it < kIters; ++it) q.memcpy(dst, src, bytes);
  q.wait();
  double dt = (now_s() - t0) / kIters;
  std::printf("%-42s %8.3f ms   %6.1f GB/s (r+w)\n", name, dt * 1e3, 2.0 * bytes / 1e9 / dt);
}
}  // namespace

int main() {
  setvbuf(stdout, nullptr, _IONBF, 0);
  sycl::queue q{sycl::property::queue::in_order()};
  std::printf("device: %s\n", q.get_device().get_info<sycl::info::device::name>().c_str());
  const size_t bytes = kFloats * sizeof(float);

  float* sh = sycl::malloc_shared<float>(kFloats, q);
  float* dev = sycl::malloc_device<float>(kFloats, q);
  for (size_t i = 0; i < kFloats; ++i) sh[i] = (i % 7) * 0.25f;
  q.memcpy(dev, sh, bytes).wait();
  float* out_sh = sycl::malloc_shared<float>(kItems, q);
  float* out_dev = sycl::malloc_device<float>(kItems, q);
  float* sh2 = sycl::malloc_shared<float>(kFloats, q);
  float* dev2 = sycl::malloc_device<float>(kFloats, q);

  auto* sh4 = reinterpret_cast<const sycl::float4*>(sh);
  auto* dev4 = reinterpret_cast<const sycl::float4*>(dev);

  volatile float sink = 0.0f;
  std::printf("[staging ok]");
  std::printf("\n-- read-only sum, %zu MB --\n", bytes >> 20);
  run(q, "shared + scalar serial-acc (warp-like)", sh, kFloats, out_sh,
      [](const float* p, size_t b) { return sum_scalar<1>(p, b); }, kItems, bytes);
  run(q, "shared + scalar 4-way ILP", sh, kFloats, out_sh,
      [](const float* p, size_t b) { return sum_scalar<4>(p, b); }, kItems, bytes);
  run(q, "shared + float4 vec (16 loads/item)", sh, kFloats, out_sh,
      [sh4](const float*, size_t b) { return sum_vec4<16>(sh4, b / 4); }, kItems, bytes);
  run(q, "device + scalar serial-acc", dev, kFloats, out_dev,
      [](const float* p, size_t b) { return sum_scalar<1>(p, b); }, kItems, bytes);
  run(q, "device + scalar 4-way ILP", dev, kFloats, out_dev,
      [](const float* p, size_t b) { return sum_scalar<4>(p, b); }, kItems, bytes);
  run(q, "device + float4 vec (16 loads/item)", dev, kFloats, out_dev,
      [dev4](const float*, size_t b) { return sum_vec4<16>(dev4, b / 4); }, kItems, bytes);

  std::printf("\n-- memcpy --\n");
  run_copy(q, "shared -> shared", sh2, sh, bytes);
  run_copy(q, "device -> device", dev2, dev, bytes);

  sink = out_sh[0] + out_dev[0];
  (void)sink;
  sycl::free(sh, q); sycl::free(dev, q); sycl::free(sh2, q); sycl::free(dev2, q);
  sycl::free(out_sh, q); sycl::free(out_dev, q);
  return 0;
}
