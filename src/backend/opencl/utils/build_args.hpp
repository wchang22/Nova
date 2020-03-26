#ifndef OPENCL_BUILD_ARGS_HPP
#define OPENCL_BUILD_ARGS_HPP

#include <array>
#include <sstream>
#include <string>

class BuildArgs {
public:
  void add_flag(const std::string& flag) { build_args << flag << " "; }

  void add_include_dir(const std::string& dir) { build_args << "-I" << dir << " "; }

  template <typename T>
  void add_define(const std::string& key, const T& value) {
    build_args << "-D" << key << "=" << value << " ";
  }

  void add_define(const std::string& key, const std::array<float, 3>& value) {
    build_args << "-D" << key << "=(float3)(" << value[0] << "," << value[1] << "," << value[2]
               << ") ";
  }

  std::string str() const { return build_args.str(); }

private:
  std::stringstream build_args;
};

#endif // OPENCL_BUILD_ARGS_HPP
