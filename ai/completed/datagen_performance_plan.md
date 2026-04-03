# Plan: Datagen Performance Optimizations

> **Purpose:** Eliminate GPU starvation and CPU bottlenecks in `monti_datagen` that
> result in ~15-20% GPU utilization, ~40% CPU utilization, and ~0% disk utilization
> when running 8+ concurrent jobs — far below hardware capacity.
>
> **Prerequisites:** Completed training data generation run (current pipeline functional).
>
> **Relationship to existing plans:** Independent of [ml_denoiser_plan.md](../completed/ml_denoiser_plan.md).
> Changes to `capture/` affect only the datagen write path and the training
> `DataLoader` read path — no impact on real-time rendering or denoising.
>
> **Session sizing:** 5 phases, each scoped for a single Copilot session.

---

## Root Cause Analysis

The primary bottleneck is **GPU-CPU synchronization overhead** in the reference
accumulation loop. For each of 256 reference frames, the current code:

1. Allocates a command buffer, records `RenderFrame`, submits, creates a fence, waits, destroys the fence, frees the command buffer.
2. Allocates a staging buffer via VMA, transitions the diffuse image to `TRANSFER_SRC`, copies to staging, transitions back, submits+waits, maps, accumulates FP16→FP32 on CPU, unmaps, destroys staging buffer.
3. Repeats step 2 for specular.

This produces **768 GPU submissions with fence waits** per viewpoint (256 × 3),
plus **512 VMA staging buffer allocations/frees**. The GPU idles during CPU
readback and accumulation; the CPU idles during GPU rendering.

Secondary bottlenecks:
- **7 individual readback submissions** for the noisy frame's G-buffer channels (should be 1).
- **CPU-side FP16→FP32 accumulation**: ~1 billion `glm::unpackHalf1x16` calls per viewpoint.
- **ZIP compression** of noisy EXR data that is essentially random (achieves ~1.0-1.3× on noisy FP16 radiance).
- **No overlap** between EXR writing and GPU rendering of the next viewpoint.

---

## Phase D1: Uncompressed EXR Default + CLI Compression Option

> **Scope:** Change EXR compression default to `NONE` for datagen. Add `--exr-compression`
> CLI option to `monti_datagen`. Update `generate_training_data.py` to pass through
> the option. Update `_GB_PER_PAIR` estimate.

### Rationale

ZIP compression on noisy FP16 path-traced radiance achieves only ~1.0-1.3× reduction
while consuming significant CPU time across 8+ concurrent processes. Smooth auxiliary
channels (albedo, normals, depth) compress better (~2-4×) but are a small fraction of
total data. Overall blended compression is ~1.3-1.8×, saving ~10-15 MB per pair at
substantial CPU cost.

Uncompressed EXR also benefits the training `DataLoader` by eliminating decompression
on every sample load.

### Changes

**`capture/include/monti/capture/Writer.h`**

Add `compression` field to `WriterDesc`:

```cpp
enum class ExrCompression {
    kNone,
    kZip,
};

struct WriterDesc {
    // ... existing fields ...
    ExrCompression compression = ExrCompression::kNone;
};
```

**`capture/src/Writer.cpp`**

- Map `ExrCompression` to tinyexr constants in `WriteExr`:

| `ExrCompression` | tinyexr constant |
|---|---|
| `kNone` | `TINYEXR_COMPRESSIONTYPE_NONE` |
| `kZip` | `TINYEXR_COMPRESSIONTYPE_ZIP` |

- Replace the hardcoded `header.compression_type = TINYEXR_COMPRESSIONTYPE_ZIP` with
  a value derived from a new `compression_type` parameter passed to `WriteExr` from
  `WriteFrame` / `WriteFrameRaw`, sourced from the `Writer` instance's stored
  compression setting.

**`app/datagen/main.cpp`**

Add CLI option:

```cpp
std::string compression_str = "none";
app.add_option("--exr-compression", compression_str,
               "EXR compression: none (default), zip")
    ->check(CLI::IsMember({"none", "zip"}));
```

Map to `ExrCompression` enum and pass through `WriterDesc`.

**`training/scripts/generate_training_data.py`**

- Add `--exr-compression` argument (default `"none"`), pass through to
  `monti_datagen` invocations.
- Update `_GB_PER_PAIR` from `0.15` to `0.04` (uncompressed: ~40 MB/pair = 0.04 GB).

**Tests**

- Update any existing Writer tests to cover both compression modes.
- Verify round-trip: write uncompressed EXR, read back with tinyexr, compare pixel data.

### Files Changed

| File | Change |
|---|---|
| `capture/include/monti/capture/Writer.h` | Add `ExrCompression` enum, add field to `WriterDesc` |
| `capture/src/Writer.cpp` | Plumb compression setting through to `WriteExr` |
| `app/datagen/main.cpp` | Add `--exr-compression` CLI option |
| `training/scripts/generate_training_data.py` | Pass `--exr-compression`, update `_GB_PER_PAIR` |

---

## Phase D2: Batch Noisy-Frame Readback into Single Submission

> **Scope:** Read back all 7 G-buffer channels in a single command buffer submission
> instead of 7 separate submissions. Reuse staging buffers across viewpoints.

### Current Problem

`RenderAndReadbackNoisy` calls `ReadbackImage` 7 times. Each call allocates a
command buffer, records barriers+copy, submits with a fence, waits, and destroys
the fence. The staging buffer is also allocated and freed each time.

This produces **7 fence-wait sync points** per viewpoint for the noisy frame alone,
plus 7 VMA staging buffer create/destroy cycles.

### Approach

#### A. Add `ReadbackMultipleImages` to `GpuReadback`

New function that accepts a span of image readback requests and executes them all in
a single command buffer:

```cpp
struct ReadbackRequest {
    VkImage image;
    uint32_t width;
    uint32_t height;
    VkDeviceSize pixel_size;
    VkPipelineStageFlags2 src_stage;
    VkPipelineStageFlags2 dst_stage;
};

// Reads back multiple images in a single command buffer submission.
// Returns one StagingBuffer per request (same order).
std::vector<StagingBuffer> ReadbackMultipleImages(
    const ReadbackContext& ctx,
    std::span<const ReadbackRequest> requests);
```

Implementation:
1. Allocate all staging buffers up front.
2. Begin one command buffer.
3. For each request: record GENERAL→TRANSFER_SRC barrier, copy, TRANSFER_SRC→GENERAL barrier.
4. Submit once, wait once.
5. Return the staging buffers.

#### B. Pre-allocate staging buffers (optional enhancement)

Add a `StagingBufferPool` that pre-allocates the 7 staging buffers once in the
`GenerationSession` constructor and reuses them across viewpoints. This avoids
VMA alloc/free per viewpoint.

```cpp
class StagingBufferPool {
public:
    StagingBufferPool(VmaAllocator allocator, std::span<const VkDeviceSize> sizes);
    std::span<StagingBuffer> Buffers();
};
```

#### C. Refactor `RenderAndReadbackNoisy`

Replace 7 individual `ReadbackImage` calls with a single `ReadbackMultipleImages`
call using the 7 G-buffer images.

### Files Changed

| File | Change |
|---|---|
| `capture/include/monti/capture/GpuReadback.h` | Add `ReadbackRequest`, `ReadbackMultipleImages` |
| `capture/src/GpuReadback.cpp` | Implement `ReadbackMultipleImages` |
| `app/datagen/GenerationSession.h` | Replace per-channel staging with pooled buffers (optional) |
| `app/datagen/GenerationSession.cpp` | Use `ReadbackMultipleImages` in `RenderAndReadbackNoisy` |

---

## Phase D3: GPU-Side Reference Accumulation via Compute Shader

> **Scope:** Replace the CPU-side 256-frame readback-and-accumulate loop with a GPU
> compute shader that accumulates directly into FP32 storage images. Read back once
> at the end.

### Current Problem

`AccumulateFrames` renders 256 frames with this per-frame loop:
1. Begin command buffer → render → submit+wait (fence)
2. `ReadbackImage` diffuse → submit+wait (fence) → map → CPU accumulate → unmap → destroy staging
3. `ReadbackImage` specular → submit+wait (fence) → map → CPU accumulate → unmap → destroy staging

Total: **768 fence waits**, **512 staging buffer allocs**, **~1 billion FP16→FP32 CPU conversions**.

### Approach

#### A. Create accumulation compute shader

New file: `shaders/accumulate.comp`

```glsl
#version 460
layout(local_size_x = 16, local_size_y = 16) in;

layout(set = 0, binding = 0, rgba16f) uniform readonly image2D u_source;
layout(set = 0, binding = 1, rgba32f) uniform image2D u_accumulator;

void main() {
    ivec2 px = ivec2(gl_GlobalInvocationID.xy);
    ivec2 size = imageSize(u_source);
    if (px.x >= size.x || px.y >= size.y) return;

    vec4 sample_val = imageLoad(u_source, px);
    vec4 accum = imageLoad(u_accumulator, px);
    imageStore(u_accumulator, px, accum + sample_val);
}
```

Add to CMake shader compilation list.

#### B. Create `GpuAccumulator` class in `capture/`

```cpp
class GpuAccumulator {
public:
    struct Desc {
        VkDevice device;
        VmaAllocator allocator;
        VkQueue queue;
        uint32_t queue_family_index;
        uint32_t width;
        uint32_t height;
        std::string_view shader_dir;
        // Vulkan function pointers (for VK_NO_PROTOTYPES)
    };

    static std::unique_ptr<GpuAccumulator> Create(const Desc& desc);

    // Clear accumulators to zero.
    void Reset(VkCommandBuffer cmd);

    // Add source images to accumulators. Called within an active command buffer.
    void Accumulate(VkCommandBuffer cmd,
                    VkImage noisy_diffuse,
                    VkImage noisy_specular);

    // Finalize: divide by frame count and read back to CPU.
    MultiFrameResult Finalize(const ReadbackContext& ctx, uint32_t num_frames);

private:
    // FP32 accumulation images (RGBA32F, VK_IMAGE_USAGE_STORAGE_BIT | TRANSFER_SRC_BIT)
    VkImage accum_diffuse_;
    VkImage accum_specular_;
    VmaAllocation accum_diffuse_alloc_;
    VmaAllocation accum_specular_alloc_;

    // Compute pipeline
    VkPipeline pipeline_;
    VkPipelineLayout pipeline_layout_;
    VkDescriptorSetLayout desc_set_layout_;
    VkDescriptorPool desc_pool_;

    // Two descriptor sets: one for diffuse source→accum, one for specular
    VkDescriptorSet diffuse_desc_set_;
    VkDescriptorSet specular_desc_set_;

    uint32_t width_, height_;
    // ... device, allocator references ...
};
```

#### C. Create `AccumulateFramesGpu` replacement

New function or method that replaces `AccumulateFrames`:

```cpp
MultiFrameResult AccumulateFramesGpu(
    GpuAccumulator& accumulator,
    const ReadbackContext& readback_ctx,
    vulkan::Renderer& renderer,
    GBufferImages& gbuffer,
    uint32_t num_frames,
    uint32_t frame_index_offset);
```

Loop structure (all on GPU until final readback):

```
accumulator.Reset(cmd)
submit+wait  // single reset
for frame in 0..num_frames:
    begin cmd
    renderer.RenderFrame(cmd, gbuffer, frame_index)
    barrier: RT → compute
    accumulator.Accumulate(cmd, diffuse_image, specular_image)
    barrier: compute → RT  // for next frame's render
    submit+wait  // ONE sync per frame instead of THREE
result = accumulator.Finalize(readback_ctx, num_frames)  // single readback
```

This reduces sync points from **768 → 257** (256 render+accumulate submissions + 1 finalize readback). The render and accumulate run in the same command buffer per frame — no readback between them.

#### D. Descriptor set management for varying source images

The accumulate shader reads from `noisy_diffuse` and `noisy_specular` images whose
handles are fixed for the lifetime of a `GBufferImages` instance. Descriptor sets
can be written once in `Create()` or when the GBuffer images change, rather than
per-frame. The `Accumulate` call dispatches twice (once per channel pair) using
pre-built descriptor sets.

Alternatively, use a single descriptor set with 4 bindings (2 sources + 2 accumulators)
and dispatch once, accumulating both channels simultaneously:

```glsl
layout(set = 0, binding = 0, rgba16f) uniform readonly image2D u_noisy_diffuse;
layout(set = 0, binding = 1, rgba16f) uniform readonly image2D u_noisy_specular;
layout(set = 0, binding = 2, rgba32f) uniform image2D u_accum_diffuse;
layout(set = 0, binding = 3, rgba32f) uniform image2D u_accum_specular;
```

This halves the dispatches.

#### E. Finalize with division

`Finalize` dispatches a second compute pass (or reuses the same shader with a push
constant flag) to divide accumulators by `num_frames`, then reads back 2 FP32 images.

Simpler alternative: skip the GPU divide, read back the raw sums, and divide CPU-side
in `WriteFrame` (the data is FP32 at that point, so division is trivial and fast).

### Files Changed

| File | Change |
|---|---|
| `shaders/accumulate.comp` | **New** — accumulation compute shader |
| `CMakeLists.txt` | Add `accumulate.comp` to shader compilation |
| `capture/include/monti/capture/GpuAccumulator.h` | **New** — `GpuAccumulator` class |
| `capture/src/GpuAccumulator.cpp` | **New** — implementation |
| `capture/include/monti/capture/GpuReadback.h` | Keep `AccumulateFrames` for backward compat (or remove if unused) |
| `app/datagen/GenerationSession.h` | Add `GpuAccumulator` member |
| `app/datagen/GenerationSession.cpp` | Use `GpuAccumulator` in `RenderReference` |

### GPU Memory Impact

Two RGBA32F images at 960×540: 2 × 960 × 540 × 16 bytes = **15.8 MB**. Negligible
relative to the 8 GB currently used.

---

## Phase D4a: Structured Timing Output from `monti_datagen`

> **Scope:** Instrument the entire `monti_datagen` lifecycle with CPU-side timing.
> Write a machine-readable `timing.json` alongside the output. Print human-readable
> timing to stderr (which is captured by the Python orchestrator on failure).

### Problem

Currently `monti_datagen` prints viewpoint progress to stdout, which
`generate_training_data.py` captures via `capture_output=True` and **discards
on success**. There is no timing data collected at any level.

### Instrumentation Points

Measure wall-clock time (`std::chrono::steady_clock`) around every major phase:

| Phase | Location | Key |
|---|---|---|
| Vulkan init | `main.cpp` — `CreateInstance` + `CreateDevice` | `vulkan_init_ms` |
| Scene load | `main.cpp` — `LoadGltf` | `scene_load_ms` |
| Environment/lights load | `main.cpp` — env map + area lights | `env_load_ms` |
| Renderer create | `main.cpp` — `Renderer::Create` | `renderer_create_ms` |
| Mesh upload + BLAS | `main.cpp` — `UploadAndRegisterMeshes` | `mesh_upload_ms` |
| G-buffer create | `main.cpp` — `GBufferImages::Create` | `gbuffer_create_ms` |
| **Per-viewpoint** (array): | | |
| &emsp;Noisy render | `GenerationSession` — `RenderFrame` call | `render_noisy_ms` |
| &emsp;Noisy readback | `GenerationSession` — all `ReadbackImage` calls | `readback_noisy_ms` |
| &emsp;Reference render+accum | `GenerationSession` — `RenderReference` | `render_reference_ms` |
| &emsp;EXR write | `GenerationSession` — `WriteFrame` | `write_exr_ms` |
| &emsp;Total | Sum of above | `total_ms` |
| **Overall total** | `main.cpp` — full program | `total_ms` |

### `timing.json` Schema

Written to `{output_dir}/timing.json` at the end of a successful run:

```json
{
  "version": 1,
  "device": "NVIDIA GeForce RTX 4090",
  "resolution": [960, 540],
  "spp": 4,
  "ref_frames": 256,
  "exr_compression": "none",
  "setup": {
    "vulkan_init_ms": 120.5,
    "scene_load_ms": 340.2,
    "env_load_ms": 45.1,
    "renderer_create_ms": 85.3,
    "mesh_upload_ms": 210.7,
    "gbuffer_create_ms": 5.2
  },
  "viewpoints": [
    {
      "index": 0,
      "render_noisy_ms": 8.2,
      "readback_noisy_ms": 3.1,
      "render_reference_ms": 1842.5,
      "write_exr_ms": 45.3,
      "total_ms": 1899.1
    }
  ],
  "summary": {
    "num_viewpoints": 12,
    "total_ms": 24500.0,
    "setup_ms": 807.0,
    "avg_viewpoint_ms": 1974.4,
    "avg_render_reference_ms": 1850.2,
    "avg_write_exr_ms": 42.1
  }
}
```

### Implementation

#### A. Timing helper

Minimal RAII timer class (local to datagen, not a library type):

```cpp
struct TimingEntry {
    std::string key;
    double ms;
};

class ScopedTimer {
public:
    explicit ScopedTimer(std::string key, std::vector<TimingEntry>& out);
    ~ScopedTimer();  // records elapsed to out
};
```

Or simpler: just capture `steady_clock::now()` pairs inline and compute durations
when building the JSON. Prefer the simpler approach — avoid adding infrastructure
for a diagnostic feature.

#### B. `main.cpp` changes

Wrap each setup phase in timing measurements. After `session.Run()`, collect the
session's per-viewpoint timings and merge with setup timings. Write `timing.json`
using `nlohmann::json` (already a dependency).

Human-readable summary also printed to stdout (alongside existing output):

```
Setup timing:
  Vulkan init:      120.5ms
  Scene load:       340.2ms
  Environment:      45.1ms
  Renderer create:  85.3ms
  Mesh upload:      210.7ms
  G-buffer create:  5.2ms
  Setup total:      807.0ms
```

#### C. `GenerationSession` changes

- Add `std::vector<nlohmann::json> viewpoint_timings_` member.
- In `Run()`, measure each phase per viewpoint and append to the vector.
- Add getter: `const std::vector<nlohmann::json>& ViewpointTimings() const`.
- Existing per-viewpoint stdout output (position, timing) is kept; the new
  structured data supplements it.

Human-readable per-viewpoint output enhanced to:

```
[viewpoint 1/12] pos=(1.2, 3.4, 5.6) target=(0.0, 1.0, 0.0) fov=45.0
  render noisy:      8.2ms
  readback noisy:    3.1ms
  render reference:  1842.5ms (256 frames, avg 7.2ms/frame)
  write EXR:         45.3ms
  total:             1899.1ms
```

### Files Changed

| File | Change |
|---|---|
| `app/datagen/main.cpp` | Wrap setup phases in timing, write `timing.json` |
| `app/datagen/GenerationSession.h` | Add `viewpoint_timings_` vector, getter |
| `app/datagen/GenerationSession.cpp` | Measure per-viewpoint phases, populate timings |

---

## Phase D4b: Timing Aggregation in `generate_training_data.py`

> **Scope:** Collect `timing.json` from each `monti_datagen` invocation and present
> aggregated timing information: live progress with throughput, and a final summary
> report.

### Problem

`generate_training_data.py` uses `subprocess.run(cmd, capture_output=True)` and
discards stdout/stderr on success. With 8+ concurrent jobs, there is no visibility
into per-job performance or overall bottleneck distribution.

### Approach

#### A. Collect `timing.json` from each invocation

In `_run_invocation`, after a successful subprocess run, read `{inv_tmp}/timing.json`
before the temp directory is cleaned up. Return the parsed timing data alongside
the success status:

```python
def _run_invocation(cmd, inv_tmp, output_dir, scene_name, group_entries):
    # ... existing subprocess.run + file move logic ...
    timing = None
    timing_path = os.path.join(inv_tmp, "timing.json")
    if os.path.isfile(timing_path):
        with open(timing_path) as f:
            timing = json.load(f)
    return True, "", timing
```

Update callers to handle the new return value.

#### B. Live progress with throughput

As each invocation completes, print a progress line with timing:

```
  [3/24] sponza (8 vp, env=studio.exr)  12.4s  avg 1.55s/vp  ref 1.31s/vp
  [4/24] bathroom (5 vp)                  7.2s  avg 1.44s/vp  ref 1.22s/vp
```

Also maintain running totals for a throughput estimate:
```
  Progress: 42/120 viewpoints  |  elapsed 01:23  |  ETA 02:55  |  3.2 vp/min
```

#### C. Final summary report

After all invocations complete, aggregate collected timing data and print:

```
=== Timing Summary ===
  Setup (avg per invocation):
    Vulkan init:      125ms
    Scene load:       380ms (range: 120-890ms)
    Mesh upload:      215ms
    Renderer create:  88ms

  Per-viewpoint averages (across 120 viewpoints):
    Render noisy:      7.8ms   (0.4% of total)
    Readback noisy:    3.2ms   (0.2%)
    Render reference:  1823ms  (92.1%)    <-- dominant bottleneck
    Write EXR:         42ms    (2.1%)
    Overhead:          105ms   (5.3%)

  Throughput:
    Total wall time:     05:32
    Viewpoints/min:      21.7
    Parallel efficiency: 68%  (8 jobs, 2.71x ideal speedup)

  Bottleneck: render_reference accounts for 92% of per-viewpoint time.
  Recommendation: Implement Phase D3 (GPU accumulation) for ~5-10x improvement.
```

#### D. Optional: write aggregate timing to file

Write `{output_dir}/generation_timing.json` containing:
- All per-invocation timing data
- Aggregate statistics
- Configuration (resolution, SPP, ref_frames, jobs, compression)

This enables tracking performance across runs and before/after optimization comparisons.

### Files Changed

| File | Change |
|---|---|
| `training/scripts/generate_training_data.py` | Collect timing.json, live progress, final summary |

---

## Phase D5: Async EXR Writing (Overlap I/O with GPU)

> **Scope:** Write EXR files on a background thread while the next viewpoint renders,
> using a double-buffer scheme.

### Current Problem

After rendering and reading back viewpoint N, the GPU sits idle while `WriteFrame`
compresses and writes EXR files to disk. With uncompressed EXR (Phase D1) this is
fast (~1-5ms), so this optimization has lower priority and smaller impact after
Phase D1. However, if ZIP compression is enabled, write time can be 50-200ms.

### Approach

Use `std::async` with a captured copy of the write data:

```cpp
// In Run() loop:
if (write_future_.valid())
    write_future_.wait();  // wait for previous viewpoint's write

// Move current frame data into a WriteJob
WriteJob job{
    .noisy_diffuse_raw = std::move(noisy_diffuse_raw_copy),
    .noisy_specular_raw = std::move(noisy_specular_raw_copy),
    // ... all channel data ...
    .ref_result = std::move(ref_result_),
    .subdirectory = std::format("vp_{}", i),
};

write_future_ = std::async(std::launch::async, [this, job = std::move(job)] {
    return WriteFrameFromJob(job);
});

// GPU starts rendering viewpoint i+1 immediately
```

The main thread copies (or moves) the readback buffers into the job, then the
background thread handles exposure scaling, B10G11R11 unpacking, and EXR writing.

#### Double-buffering readback storage

The readback CPU buffers (`noisy_diffuse_raw_`, etc.) need double-buffering:
one set for the in-flight write job, one set for the current render. Use two
sets of buffers indexed by `frame_parity = i % 2`, or use `std::vector::swap`
to transfer ownership to the async job.

The simpler approach: `std::move` the vectors into the job struct, then
re-allocate (or `.resize()`) fresh vectors for the next frame. The re-allocation
is negligible since the allocator will typically recycle the same memory.

### Files Changed

| File | Change |
|---|---|
| `app/datagen/GenerationSession.h` | Add `WriteJob` struct, `std::future<bool>` member |
| `app/datagen/GenerationSession.cpp` | Async write dispatch, wait-before-next-write pattern |

---

## Implementation Priority

| Priority | Phase | Impact | Effort |
|---|---|---|---|
| 1 | **D1** — Uncompressed EXR | Eliminates CPU compression contention across all jobs | Small |
| 2 | **D4a** — Structured timing in `monti_datagen` | Required to validate remaining optimizations | Small |
| 3 | **D4b** — Timing aggregation in Python | Visibility into concurrent job performance | Small |
| 4 | **D3** — GPU accumulation | Eliminates the dominant bottleneck (768→257 syncs, no CPU accumulation) | Medium |
| 5 | **D2** — Batch noisy readback | Reduces 7→1 sync points per noisy frame | Small-Medium |
| 6 | **D5** — Async EXR writing | Overlaps I/O with GPU; lower impact after D1 | Small-Medium |

### Expected Combined Improvement

| Metric | Current (est.) | After D1-D5 (est.) |
|---|---|---|
| GPU utilization (8 jobs) | 15-20% | 60-80% |
| Time per viewpoint (256 ref frames) | 5-15s | 1.5-3.0s |
| Sync points per viewpoint | 775 (768 + 7) | 258 (257 + 1) |
| Staging buffer allocs per viewpoint | 519 (512 + 7) | 9 (7 reused + 2 final) |
| CPU FP16→FP32 conversions per VP | ~1.06 billion | 0 |
| EXR disk size per pair | ~25-30 MB | ~40 MB |
| Performance visibility | None | Full: per-phase, per-VP, aggregate |

---

## Concurrent Job Tuning Guide

The optimal `--jobs` count changes as optimizations shift the bottleneck between
CPU, GPU, and I/O. The goal is full desktop utilization: GPU near 100%, CPU near
100%, and neither resource idle-waiting on the other.

### Why Utilization is Low (i9-12900K + RTX 3090/4090, 8 jobs)

Each `monti_datagen` process is **single-threaded** on the CPU side. The FP16→FP32
accumulation loop (~1 billion conversions), ZIP compression, fence waits, and
readback all execute sequentially on one thread per process.

**Observed:** 8 jobs, CPU at 40%, GPU at 17%, VRAM at 8/24 GB.

This is fully explained by the per-job execution pattern:

```
Timeline for ONE job (single viewpoint, ~5-15s):

CPU thread:  [submit]---idle---[readback+accumulate 100ms]---[submit]---idle---...
GPU queue:   ........[RT 7ms].............................[RT 7ms]...........

Over 256 reference frames, the GPU does 256 × 7ms = ~1.8s of actual work,
but the CPU stalls between each submission add ~100ms × 256 = ~25s of gaps.
```

With 8 concurrent jobs, you get 8 CPU threads and 8 interleaved GPU bursts:

```
i9-12900K:  20 logical CPUs → 8 busy / 20 total = 40% CPU utilization  ✓
GPU:        8 jobs × 7ms bursts with ~100ms gaps → ~17% GPU utilization  ✓
VRAM:       ~1 GB/job × 8 = 8 GB / 24 GB = 33%                          ✓
```

**Each resource is underutilized for a different reason:**
- **CPU at 40%:** Only 8 of 20 logical CPUs are active. 12 sit idle.
- **GPU at 17%:** Even with 8 jobs, the GPU bursts are too short relative to
  CPU stall time. The GPU finishes its 7ms of RT and waits ~100ms for the
  next submission from that job.
- **VRAM at 33%:** Simply 8 × 1 GB. Not a constraint.

**The fix for the current (unoptimized) code: more jobs.** Each additional job
adds one CPU thread and slightly more GPU overlap. With 20 logical CPUs and
24 GB VRAM, you can safely run 16-18 jobs right now.

### Recommended Job Counts by Phase

Hardware reference: i9-12900K (8P + 8E = 20 logical CPUs), 24 GB VRAM.

| Optimization State | `--jobs` | CPU util | GPU util | Reasoning |
|---|---|---|---|---|
| **Baseline** (current) | **16-18** | 80-90% | 25-35% | Each job = 1 CPU thread. Fill all logical CPUs. GPU still starved but fed more often. VRAM: 16-18 GB, fits. |
| **After D1** (uncompressed EXR) | **16-20** | 80-100% | 30-40% | ZIP compression eliminated → CPU per-job load drops. Can push to 20 jobs = 20 threads = 100% CPU. |
| **After D1 + D2** (batch readback) | **16-20** | 80-100% | 30-40% | CPU still dominates (FP16 accumulation). Similar to D1-only. |
| **After D1 + D3** (GPU accumulation) | **3-6** | 30-50% | 70-95% | GPU becomes the bottleneck. Each job now keeps the GPU busy ~90% of its time slot. Fewer jobs to avoid queue contention. |
| **After D1-D5** (all optimizations) | **3-6** | 30-50% | 75-95% | GPU-bound. Target GPU at 85-95%. CPU has headroom. |

**Key insight:** Before D3, the system is CPU-bound — add jobs until CPU is
at 80-90%. After D3, the system is GPU-bound — reduce jobs until GPU is at
85-95%. The crossover happens at D3 (GPU accumulation).

### How to Use D4a/D4b Timing Data

After implementing D4a (structured timing in `monti_datagen`) and D4b (aggregation
in `generate_training_data.py`), the final summary report provides the data needed
to select the right job count. Run the same workload at several `--jobs` values
(e.g., 2, 4, 8, 12) and compare:

#### Step 1: Identify the per-job bottleneck

From the per-viewpoint timing breakdown:

```
Per-viewpoint averages (across 120 viewpoints):
  Render noisy:      7.8ms   (0.4% of total)
  Readback noisy:    3.2ms   (0.2%)
  Render reference:  1823ms  (92.1%)    <-- GPU-bound
  Write EXR:         42ms    (2.1%)     <-- I/O-bound
  Overhead:          105ms   (5.3%)     <-- CPU-bound (sync, alloc)
```

- If `render_reference` dominates: **GPU-bound** — fewer jobs, each gets more GPU time.
- If `write_exr` dominates: **I/O-bound** — likely disk bandwidth limited (unlikely with NVMe + uncompressed).
- If `overhead` is large relative to render: **CPU-bound** from sync/alloc — more jobs to keep GPU fed.
- If `readback_noisy` is surprisingly large: **memory bandwidth-bound** — benchmark staging buffer reuse.

#### Step 2: Check parallel efficiency

The summary report includes a parallel efficiency metric:

```
Throughput:
  Total wall time:     05:32
  Viewpoints/min:      21.7
  Parallel efficiency: 68%  (8 jobs, 2.71x ideal speedup)
```

- **Parallel efficiency** = (single-job throughput × jobs) / actual throughput.
- If efficiency drops sharply when adding jobs (e.g., 8 jobs but only 2.7x
  speedup → 34% efficiency), you have contention — **reduce jobs**.
- If efficiency stays high (e.g., 8 jobs, 6.5x speedup → 81%), there is
  headroom — **try more jobs**.

To calculate: run once with `--jobs 1` to establish single-job baseline
throughput, then compare against multi-job runs.

#### Step 3: Cross-reference with system metrics

While a generation run proceeds, observe in Task Manager or GPU-Z:

| Metric | Underpowered (add jobs) | Sweet spot | Contention (reduce jobs) |
|---|---|---|---|
| GPU 3D utilization | < 60% | 75-95% | 95-100% with frame drops |
| GPU VRAM usage | < 50% of total | 50-80% | > 90% (risk of OOM) |
| CPU total utilization | < 50% | 60-90% | 100% sustained (thermal throttle risk) |
| Disk write throughput | < 100 MB/s | Matches expected output rate | Queue depth spikes |

#### Step 4: Iterate

1. Start with the recommended `--jobs` from the table above.
2. Run a small batch (e.g., 20 viewpoints) at that count.
3. Check the D4b summary: efficiency, bottleneck distribution, wall time.
4. Adjust ±2 jobs and re-run. Optimum is found when:
   - GPU utilization is 75-95%.
   - Parallel efficiency is > 60%.
   - Per-viewpoint time does not increase significantly vs. fewer jobs.
5. Record the sweet spot in `generation_timing.json` for future reference.

### VRAM Budget (i9-12900K, 24 GB VRAM)

Observed: ~1 GB per `monti_datagen` process (8 jobs = 8 GB VRAM).

| Jobs | VRAM (est.) | Headroom | Notes |
|---|---|---|---|
| 8 | 8 GB | 16 GB free | Current. Severely underutilized. |
| 12 | 12 GB | 12 GB free | Comfortable. |
| 16 | 16 GB | 8 GB free | Good. Leaves room for OS + desktop. |
| 18 | 18 GB | 6 GB free | Near max for pre-D3. Monitor in Task Manager. |
| 20+ | 20+ GB | < 4 GB | Risk of VRAM pressure. Only with small scenes. |

After D3 (GPU accumulation), per-process VRAM increases slightly (~20 MB for
accumulation buffers), but the job count drops to 3-6, so total VRAM usage
actually decreases (3-6 GB).

### Quick-Reference Decision Tree

```
Is GPU utilization > 90%?
  YES → Reduce --jobs by 2. Re-check.
  NO  →
    Is GPU utilization < 60%?
      YES → Increase --jobs by 2. Re-check.
      NO  →
        Is parallel efficiency > 60%?
          YES → Current --jobs is near optimal.
          NO  → CPU or memory contention. Profile D4a overhead breakdown.
```
