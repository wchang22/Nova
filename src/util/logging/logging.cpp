#include "logging.hpp"

#include "util/exception/exception.hpp"

#if __has_include(<filesystem>)
  #include <filesystem>
namespace fs = std::filesystem;
#elif __has_include(<experimental/filesystem>)
  #include <experimental/filesystem>
namespace fs = std::experimental::filesystem;
#endif

logger_t Logging::logger = nullptr;

logger_t Logging::get_logger() {
  if (logger) {
    return logger;
  }

#ifdef LOG
  fs::create_directory("logs");
  logger = std::make_shared<std::ofstream>("logs/debug.log");

  if (!logger->is_open()) {
    throw LoggingException("Cannot open logging file logs/debug.log");
  }
#endif

  return logger;
}
