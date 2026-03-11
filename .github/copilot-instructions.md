## Coding Guidelines

### C++ Naming Conventions (Google C++ Style)
*   **Files:** Use `PascalCase` for files named after their primary class or type (e.g., `GpuScene.cpp`, `Renderer.h`, `BlueNoise.h`). Use `snake_case` for application entry points, test files, and multi-purpose utility files (e.g., `vulkan_context.cpp`, `gpu_scene_test.cpp`, `main.cpp`). Header files use `.h`, source files use `.cpp`.
*   **Types:** Use `PascalCase` for classes, structs, enums, type aliases, and template parameters (e.g., `VulkanContext`, `PathTracer`, `DenoiserInput`).
*   **Functions and Methods:** Use `PascalCase` for free functions and methods (e.g., `Initialize()`, `TraceRays()`, `UpdateScene()`).
*   **Variables:** Use `snake_case` for local variables and function parameters (e.g., `frame_index`, `max_bounces`, `render_width`).
*   **Member Variables:** Use `snake_case` with a trailing underscore for class member fields (e.g., `device_`, `swapchain_`, `frame_index_`).
*   **Constants:** Use `kPascalCase` for compile-time constants (e.g., `kMaxBounces`, `kBlueNoiseTableSize`).
*   **Enum Values:** Use `kPascalCase` (e.g., `kStatic`, `kDynamic`, `kKinematic`).
*   **Namespaces:** Use `snake_case` for namespaces (e.g., `chessboard`, `render`, `physics`).
*   **Macros:** Use `ALL_CAPS_SNAKE_CASE` for macros. Avoid macros where possible.

### Modern C++ (C++20 Required)
*   **Standard:** Use C++20 throughout. Always prefer modern C++ idioms and standard library facilities over legacy patterns.
*   **Ownership:** Use `std::unique_ptr` for exclusive ownership and `std::shared_ptr` only when shared ownership is genuinely required. NEVER use `new` or `delete` directly — all heap allocations must go through smart pointers or RAII containers.
*   **Views and References:** Prefer `std::string_view` over `const std::string&` for read-only string parameters. Prefer `std::span<T>` over raw pointer + size pairs for contiguous ranges. Prefer references over pointers for non-nullable parameters. Raw pointers indicate non-owning, non-nullable references only.
*   **Value Semantics:** Use `std::optional` for values that may be absent. Use `std::variant` for type-safe unions. Use `std::expected` (or return codes where `std::expected` is unavailable) for fallible operations.
*   **Move Semantics:** Prefer move semantics over copies for non-trivial types. Use `std::move` explicitly when transferring ownership. Implement move constructors and move assignment operators for resource-owning classes.
*   **Type Inference:** Use `auto` for local variable declarations when the type is clear from the initializer or when the exact type is unimportant. Do not use `auto` when it obscures the type in a way that harms readability.
*   **Iteration:** Use range-based `for` loops over index-based loops whenever iterating over containers or ranges. Use `std::ranges` algorithms where they improve clarity.
*   **Concurrency:** Use `std::future<T>`, `std::async`, and coroutines (`co_await`, `co_yield`) for asynchronous work. Avoid raw threads with manual synchronization where higher-level abstractions suffice.
*   **Scoped Enums:** Always use `enum class` over unscoped `enum`.
*   **Structured Bindings:** Use structured bindings (`auto [key, value] = ...`) when destructuring pairs, tuples, or structs.
*   **Constexpr:** Prefer `constexpr` over `const` for compile-time constants. Use `consteval` for functions that must be evaluated at compile time.
*   **RAII:** Prefer RAII wrappers for Vulkan objects and other resources that require explicit cleanup. No manual resource management outside of RAII destructors.
*   **Avoid Legacy Patterns:** Do not use C-style casts (use `static_cast`, `reinterpret_cast`), C-style arrays (use `std::array` or `std::vector`), `NULL` (use `nullptr`), or `typedef` (use `using`).

### Shading Languages
*   **Semicolons:** ALWAYS use semicolons at the end of statements in C++, GLSL, and all shading languages.
*   **Floating Point Literals:** ALWAYS use `.0` to denote floating-point literals for whole numbers (e.g., `1.0`, `0.0`) and decimal notation for fractional numbers (e.g., `0.5`, `1.23`). This ensures type correctness, especially in shading languages.

### Headers and Includes
*   **Headers:** Use `#pragma once` for header guards.
*   **Includes:** Order includes: corresponding header first, then C++ standard library, then third-party libraries, then project headers. Separate groups with a blank line.

### Error Handling
*   **Error Handling:** Use `std::optional` or `std::expected` for expected failures. Reserve exceptions for truly exceptional conditions. For Vulkan calls, check `VkResult` and handle errors explicitly.

### General Coding Style
*   Prioritize required fields and function arguments over optional ones or default parameters. Only make a field optional if it has a use case, or if initialization complications require it.
*   Keep class member fields private by default. Add getters sparingly, only when external access is essential. Prefer internal methods for managing private field state. Add setters only as a last resort when direct modification is the clearest approach.
*   Use single-line conditionals (without braces) for a single statement if it fits within 100 columns.
*   Use single-line conditionals (without braces) for `return`, `continue`, and `throw` statements.
    *   Consider restructuring code to enable single-line conditionals where it improves readability.
*   Prefer minimal changes.
*   Prefer mandatory arguments unless the code requires an optional argument.
*   Prefer to implement only what is needed, and avoid adding unused features.
*   Prefer minimal changes to variable, function, and class names.
*   Avoid removing existing comments unless they are incorrect or misleading.
*   If the function or variable name is clear, avoid adding comments that simply restate the name.

### When working in PathTracer or Denoiser implementations
*   The `PathTracer` and `Denoiser` are abstract interfaces. Implementations must not leak backend-specific types through the interface.
*   Output images from `PathTracer` are consumed directly by `Denoiser`. Coordinate image formats and layouts between them via the interface contracts, not via direct coupling.
*   When modifying shaders, do not remove existing functionality unless specifically requested. Clearly explain any refactoring or substitution of new functionality.
*   When modifying shared GLSL includes (`shaders/include/`), ensure changes are compatible with all shader stages that include them (raygen, closest-hit, miss, compute).
*   When changing uniforms or push constants, ensure proper alignment to `std140`/`std430` rules and update both the GLSL struct and the C++ struct that maps to it.