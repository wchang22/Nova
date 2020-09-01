#include "logging.hpp"

#include "util/exception/exception.hpp"

logger_t Logging::logger = nullptr;

logger_t Logging::get_logger() {
  if (logger) {
    return logger;
  }

#ifdef LOG
  logger = std::make_shared<std::ofstream>("logs/debug.log");

  if (!logger->is_open()) {
    throw LoggingException("Cannot open logging file logs/debug.log");
  }
#endif

  return logger;
}
