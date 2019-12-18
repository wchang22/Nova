#ifndef CAMERA_H
#define CAMERA_H

#include <CL/cl2.hpp>
#include <glm/glm.hpp>

using namespace glm;

class Camera {
public:
  Camera(const vec3& position, const vec3& forward, const vec3& up,
         int width, int height, float fovy);

  struct EyeCoords {
    cl_float2 coord_scale;
    cl_float2 coord_dims;
    cl_float3 eye_pos;
    cl_float3 eye_coord_frame0;
    cl_float3 eye_coord_frame1;
    cl_float3 eye_coord_frame2;
  };

  EyeCoords get_eye_coords() const;

private:
  vec3 position;
  vec3 forward;
  vec3 up;

  int width, height;
  float fovy;
};

#endif // CAMERA_H
