#include <stb_image.h>
#include <sstream>

#include "raytracer.h"
#include "util/file/fileutils.h"
#include "util/exception/exception.h"
#include "util/image/imageutils.h"
#include "util/profiling/profiling.h"
#include "util/kernel/kernelutils.h"
#include "util/opencl/clutils.h"
#include "constants.h"

Raytracer::Raytracer(uint32_t width, uint32_t height, const std::string& name)
  : width(width), height(height),
    name(name),
    camera_settings(scene_parser.get_camera_settings()),
    camera(camera_settings.position, camera_settings.forward, camera_settings.up,
           width, height, camera_settings.fovy),
    intersectable_manager(name),
    context(DEVICE_TYPE)
{
  const auto model_paths = scene_parser.get_model_paths();
  for (const auto& model_path : model_paths) {
    Model model(model_path, material_loader);
    intersectable_manager.add_model(model);
  }

  const auto [ default_diffuse, default_metallic, default_roughness, default_ambient_occlusion ]
    = scene_parser.get_shading_default_settings();
  const auto [ light_position, light_intensity ] = scene_parser.get_light_settings();
  const unsigned int ray_recursion_depth = scene_parser.get_ray_recursion_depth();

  cl::Device device(context.getInfo<CL_CONTEXT_DEVICES>().front());
  queue = cl::CommandQueue(context, device, CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE);
  cl::Program program(context, file_utils::read_file(KERNEL_PATH));

  try {
    std::stringstream build_args;
    build_args
      << " -cl-fast-relaxed-math -cl-mad-enable"
      << " -I" << KERNELS_PATH
      << " -D" << STRINGIFY(TRIANGLES_PER_LEAF_BITS) << "=" << TRIANGLES_PER_LEAF_BITS
      << " -DDEFAULT_DIFFUSE=" << "(float3)("
        << default_diffuse.x << "," << default_diffuse.y << "," << default_diffuse.z << ")"
      << " -DDEFAULT_METALLIC=" << default_metallic
      << " -DDEFAULT_ROUGHNESS=" << default_roughness
      << " -DDEFAULT_AMBIENT_OCCLUSION=" << default_ambient_occlusion
      << " -DLIGHT_POSITION=" << "(float3)("
        << light_position.x << "," << light_position.y << "," << light_position.z << ")"
      << " -DLIGHT_INTENSITY=" << "(float3)("
        << light_intensity.x << "," << light_intensity.y << "," << light_intensity.z << ")"
      << " -DRAY_RECURSION_DEPTH=" << ray_recursion_depth;
    program.build(build_args.str().c_str());
  } catch (...) {
    throw KernelException(program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device));
  }

  kernel = cl::Kernel(program, "raytrace");
}

void Raytracer::raytrace() {
  PROFILE_SCOPE("Raytrace");

  PROFILE_SECTION_START("Build data");
  cl::Image2D image(context, CL_MEM_WRITE_ONLY,
                    cl::ImageFormat(CL_RGBA, CL_UNSIGNED_INT8), width, height);
  std::vector<uint8_t> image_buf(width * height * STBI_rgb_alpha);

  auto ec = camera.get_eye_coords();

  auto [ triangle_data, triangle_meta_data, bvh_data ] = intersectable_manager.build();
  
  cl::Buffer triangle_buf(context, CL_MEM_COPY_HOST_PTR | CL_MEM_READ_ONLY,
                          triangle_data.size() * sizeof(decltype(triangle_data)::value_type),
                          triangle_data.data());
  cl::Buffer tri_meta_buf(context, CL_MEM_COPY_HOST_PTR | CL_MEM_READ_ONLY,
                          triangle_meta_data.size() *
                          sizeof(decltype(triangle_meta_data)::value_type),
                          triangle_meta_data.data());
  cl::Buffer bvh_buf(context, CL_MEM_COPY_HOST_PTR | CL_MEM_READ_ONLY,
                     bvh_data.size() * sizeof(decltype(bvh_data)::value_type),
                     bvh_data.data());

  // If no materials loaded, create a non-zero sized dummy array
  MaterialData material_data = material_loader.build();
  cl::Image2DArray material_ims;
  if (material_data.num_materials == 0) {
    material_ims = cl::Image2DArray(context, CL_MEM_READ_ONLY,
                                    cl::ImageFormat(CL_RGBA, CL_UNSIGNED_INT8), 1, 1, 1,
                                    0, 0, nullptr);
  } else {
    material_ims = cl::Image2DArray(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                    cl::ImageFormat(CL_RGBA, CL_UNSIGNED_INT8),
                                    material_data.num_materials,
                                    material_data.width, material_data.height,
                                    0, 0, material_data.data.data());
  }
  PROFILE_SECTION_END();

  PROFILE_SECTION_START("Raytrace profile");
  for (int i = 0; i < NUM_PROFILE_ITERATIONS; i++) {
    PROFILE_SCOPE("Raytrace profile loop");

    PROFILE_SECTION_START("Enqueue kernel");
    kernel_utils::set_args(kernel, image, ec, triangle_buf, tri_meta_buf, bvh_buf, material_ims);
    queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(width, height), cl::NullRange);
    queue.finish();
    PROFILE_SECTION_END();

    PROFILE_SECTION_START("Read image");
    queue.enqueueReadImage(image, true,
                           cl_utils::create_size_t<3>({ 0, 0, 0 }),
                           cl_utils::create_size_t<3>({ width, height, 1 }),
                           0, 0, image_buf.data());
    PROFILE_SECTION_END();
  }
  PROFILE_SECTION_END();

  PROFILE_SECTION_START("Write image");
  image_utils::write_image((name + ".jpg").c_str(), { image_buf, width, height });
  PROFILE_SECTION_END();
}
