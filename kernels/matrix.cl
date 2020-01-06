#ifndef MATRIX_CL
#define MATRIX_CL

typedef struct {
  float3 x;
  float3 y;
  float3 z;
} Mat3x3;

typedef struct {
  float4 x;
  float4 y;
  float4 z;
} Mat4x3;

Mat3x3 mat4x3_to_mat3x3(Mat4x3 mat) {
  return (Mat3x3) {
    mat.x.xyz,
    mat.y.xyz,
    mat.z.xyz
  };
}

float3 mat3x3_vec3_mult(Mat3x3 mat, float3 vec) {
  return (float3)(
    dot(mat.x, vec),
    dot(mat.y, vec),
    dot(mat.z, vec)
  );
}

Mat3x3 mat3x3_transpose(Mat3x3 mat) {
  return (Mat3x3) {
    (float3)(mat.x.x, mat.y.x, mat.z.x),
    (float3)(mat.x.y, mat.y.y, mat.z.y),
    (float3)(mat.x.z, mat.y.z, mat.z.z)
  };
}

float3 mat4x3_vec3_mult(Mat4x3 mat, float3 vec) {
  float4 vec4 = (float4)(vec, 1.0f);
  return (float3)(
    dot(mat.x, vec4),
    dot(mat.y, vec4),
    dot(mat.z, vec4)
  );
}

#endif // MATRIX_CL
