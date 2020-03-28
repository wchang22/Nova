#include "fileutils.hpp"
#include "util/exception/exception.hpp"

#include <fstream>
#include <streambuf>

namespace file_utils {
std::string read_file(const char* path) {
  std::ifstream file(path);

  if (!file.is_open()) {
    throw FileException("Cannot open file " + std::string(path));
  }

  return std::string(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>());
}
}
