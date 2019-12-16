#ifndef LOGGING_H
#define LOGGING_H

#include <fstream>
#include <memory>

using logger_t = std::shared_ptr<std::ofstream>;

class Logging
{
public:
  Logging() = delete;

  static logger_t get_logger();

private:
  static logger_t logger;
};

template <typename T>
inline logger_t operator<<(logger_t logger, const T& message) {
#ifdef LOG
  *logger << message;
#else
  (void) message;
#endif
  return logger;
}

inline logger_t operator<<(logger_t logger, std::ostream&(*f)(std::ostream&)) {
#ifdef LOG
  *logger << f;
#else
  (void) f;
#endif
  return logger;
}

#endif // LOGGING_H
