// Single translation unit for VMA implementation.
// This ensures VMA symbols are available to all targets that link monti_vulkan,
// regardless of whether MONTI_BUILD_APPS is enabled.

#define VMA_IMPLEMENTATION
#define VMA_STATIC_VULKAN_FUNCTIONS 0
#define VMA_DYNAMIC_VULKAN_FUNCTIONS 1

#ifdef _MSC_VER
#pragma warning(push, 0)
#endif
#include <vk_mem_alloc.h>
#ifdef _MSC_VER
#pragma warning(pop)
#endif
