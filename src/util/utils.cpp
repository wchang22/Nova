#include "utils.h"
#include "util/exception/exception.h"

#include <fstream>
#include <streambuf>

namespace utils {
  std::string read_file(const char* path) {
    std::ifstream file(path);

    if (!file.is_open()) {
      throw FileException("Cannot open file " + std::string(path));
    }

    return std::string(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>());
  }
}
