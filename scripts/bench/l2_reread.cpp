// L2 residency probe: does re-reading a buffer of size S hit L2?
// One kernel streams the whole buffer R times (each work-item loops over its
// own region R times). Effective bandwidth = S*R*4 / time. If S <= L2 the
// repeats come out at L2 speed; past L2 capacity every repeat is DRAM.
// The solver re-reads efc_J (~15-30 MB) ~30x per solve -- the S where the
// curve drops is the budget for keeping J resident.
// Build: icx /fsycl /O2 l2_reread.cpp /Fe:l2_reread.exe
#include <sycl/sycl.hpp>
#include <chrono>
#include <cstdio>

namespace {
constexpr int kIters = 6;
constexpr int kRepeats = 8;
constexpr size_t kPerItem = 64;  // floats per work-item

double now_s() {
  return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}
}  // namespace

int main() {
  setvbuf(stdout, nullptr, _IONBF, 0);
  sycl::queue q{sycl::property::queue::in_order()};
  std::printf("device: %s\n", q.get_device().get_info<sycl::info::device::name>().c_str());
  std::printf("re-read bandwidth (%d sequential passes per kernel over S):\n", kRepeats);

  for (size_t mb : {4ull, 8ull, 12ull, 16ull, 24ull, 32ull, 64ull}) {
    const size_t n = mb * 1024 * 1024 / sizeof(float);
    const size_t items = n / kPerItem;
    float* buf = sycl::malloc_shared<float>(n, q);
    float* out = sycl::malloc_shared<float>(items, q);
    for (size_t i = 0; i < n; ++i) buf[i] = (i % 13) * 0.5f;

    // each pass streams the WHOLE buffer (items stride across all of it),
    // so pass 2..R hits cache only if S fits -- that is the residency test.
    // volatile defeats load hoisting across the repeat loop (the first run
    // without it reported 335 GB/s at S=64 MB: the compiler fused the
    // identical loads across passes).
    const volatile float* vbuf = buf;
    auto body = [=](sycl::id<1> it) {
      const size_t tid = it.get(0);
      float acc = 0.0f;
      for (int r = 0; r < kRepeats; ++r) {
        for (size_t i = tid; i < n; i += items) acc += vbuf[i];
      }
      out[tid] = acc;
    };
    for (int w = 0; w < 2; ++w) q.parallel_for(sycl::range<1>(items), body);
    q.wait();
    double t0 = now_s();
    for (int it = 0; it < kIters; ++it) q.parallel_for(sycl::range<1>(items), body);
    q.wait();
    double dt = (now_s() - t0) / kIters;
    const double bytes = double(n) * sizeof(float) * kRepeats;
    std::printf("  S=%3zu MB  %7.3f ms   %6.1f GB/s\n", mb, dt * 1e3, bytes / 1e9 / dt);
    sycl::free(buf, q);
    sycl::free(out, q);
  }
  return 0;
}
