/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#ifndef _SHADERIO_CORE_H_
#define _SHADERIO_CORE_H_

#ifndef SUBGROUP_SIZE
#define SUBGROUP_SIZE 32
#endif

#ifdef __cplusplus
namespace shaderio {
using namespace glm;
#define BUFFER_REF(refname) uint64_t

static uint32_t inline adjustClusterProperty(uint32_t in)
{
  return (in + 31) & ~31;
}

#define BUFFER_REF_DECLARE(refname, typ, keywords, alignment)                                                          \
  static_assert(alignof(typ) == alignment || (alignment > alignof(typ) && ((alignment % alignof(typ)) == 0)),          \
                "Alignment incompatible: " #refname)

#define BUFFER_REF_DECLARE_ARRAY(refname, typ, keywords, alignment)                                                    \
  static_assert(alignof(typ) == alignment || (alignment > alignof(typ) && ((alignment % alignof(typ)) == 0)),          \
                "Alignment incompatible: " #refname)

#define BUFFER_REF_DECLARE_SIZE(sizename, typ, size)                                                                   \
  static_assert(sizeof(typ) == size_t(size), "GLSL vs C++ size mismatch: " #typ)

#else  // GLSL

#extension GL_EXT_shader_explicit_arithmetic_types_int8 : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int32 : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int16 : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : enable
#extension GL_EXT_buffer_reference : enable
#extension GL_EXT_buffer_reference2 : enable
#extension GL_EXT_scalar_block_layout : enable
#extension GL_EXT_shader_atomic_int64 : enable

#define PACKED_GET(flag, cfg)   (((flag) >> (true ? cfg)) & ((1 << (false ? cfg))-1))
#define PACKED_FLAG(cfg, val)   ((val) << (true ? cfg))
#define PACKED_MASK(cfg)        (((1 << (false ? cfg))-1) << (true ? cfg))

#define BUFFER_REF(refname) refname

#define BUFFER_REF_DECLARE(refname, typ, keywords, alignment)                                                          \
  layout(buffer_reference, buffer_reference_align = alignment, scalar) keywords buffer refname                         \
  {                                                                                                                    \
    typ d;                                                                                                             \
  };

#define BUFFER_REF_DECLARE_ARRAY(refname, typ, keywords, alignment)                                                    \
  layout(buffer_reference, buffer_reference_align = alignment, scalar) keywords buffer refname                         \
  {                                                                                                                    \
    typ d[];                                                                                                           \
  };

#define BUFFER_REF_DECLARE_SIZE(sizename, typ, size) const uint32_t sizename = size

#define getWorkGroupIndexLinearized(workGroupID) (workGroupID.x + (workGroupID.y * gl_NumWorkGroups.x))

#endif

BUFFER_REF_DECLARE_ARRAY(uint8s_in, uint8_t, readonly, 1);
BUFFER_REF_DECLARE_ARRAY(uint8s_inout, uint8_t, , 1);
BUFFER_REF_DECLARE_ARRAY(uint16s_in, uint16_t, readonly, 2);
BUFFER_REF_DECLARE_ARRAY(uint16s_inout, uint16_t, , 2);
BUFFER_REF_DECLARE_ARRAY(uint32s_in, uint32_t, readonly, 4);
BUFFER_REF_DECLARE_ARRAY(uint32s_inout, uint32_t, , 4);
BUFFER_REF_DECLARE_ARRAY(uvec2s_in, uvec2, , 8);
BUFFER_REF_DECLARE_ARRAY(uvec2s_inout, uvec2, , 8);
BUFFER_REF_DECLARE_ARRAY(uint64s_in, uint64_t, readonly, 8);
BUFFER_REF_DECLARE_ARRAY(uint64s_inout, uint64_t, , 8);
BUFFER_REF_DECLARE_ARRAY(uint64s_coh, uint64_t, coherent, 8);
BUFFER_REF_DECLARE_ARRAY(uint64s_coh_volatile, uint64_t, coherent volatile, 8);
BUFFER_REF_DECLARE_ARRAY(vec2s_in, vec2, readonly, 8);
BUFFER_REF_DECLARE_ARRAY(vec3s_in, vec3, readonly, 4);
BUFFER_REF_DECLARE_ARRAY(vec3s_inout, vec3, , 4);
BUFFER_REF_DECLARE_ARRAY(vec4s_in, vec4, readonly, 16);

struct DispatchIndirectCommand
{
  uint gridX;
  uint gridY;
  uint gridZ;
};

struct DrawMeshTasksIndirectCommandNV
{
  uint count;
  uint first;
};

struct DrawMeshTasksIndirectCommandEXT
{
  uint gridX;
  uint gridY;
  uint gridZ;
};

#ifdef __cplusplus
static inline
#endif
    uvec3
    fit16bitLaunchGrid(uint count)
{
  // output grid dimensions must be <= 16 bit
  // input count typically <= 24 bits

  // keep 1D
  if(count <= 0xFFFF)
    return uvec3(count, 1, 1);

  // The bit casting here makes sure we round up in case the cast to
  // float rounded down:
  float countF = float(count);
  uint  n      = uint(ceil(uintBitsToFloat(floatBitsToUint(sqrt(countF)) + 1)));
  // Now we find the last m such that n^2 - m^2 >= count.
  // Then we can factorize the left-hand side as (n-m) * (n+m), and get an
  // error that's m^2 better.
  uint m = uint(sqrt(float(n * n - count)));

  return uvec3(n - m, n + m, 1);
}

#ifdef __cplusplus
}
#endif
#endif  // _SHADERIO_CORE_H_