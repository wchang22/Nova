#ifndef OPENCL_BUILD_ARGS_H
#define OPENCL_BUILD_ARGS_H

#include <sstream>
#include <string>
#include <glm/glm.hpp>

class BuildArgs {
public:
  void add_flag(const std::string& flag) {
    build_args << flag << " ";
  }

  void add_include_dir(const std::string& dir) {
    build_args << "-I" << dir << " ";
  }
 
  template <typename T>
  void add_define(const std::string& key, const T& value) {
    build_args << "-D" << key << "=" << value << " ";
  }

  void add_define(const std::string& key, const glm::vec3& value) {
    build_args << "-D" << key
               << "=(float3)(" << value.x << "," << value.y << "," << value.z << ") ";
  }

  std::string str() const {
    return build_args.str();
  }

private:
  std::stringstream build_args;
};

#endif // OPENCL_BUILD_ARGS_H
