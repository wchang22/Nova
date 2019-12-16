#ifndef TIMESCOPE_H
#define TIMESCOPE_H

#include <chrono>
#include <string>

namespace Profiling {
  class TimeScope {
  public:
    TimeScope(const std::string& name);
    ~TimeScope();

    void section_start(const std::string& message);
    void section_end();

  private:
    std::chrono::steady_clock::time_point start;
    std::chrono::steady_clock::time_point t0;
    std::string message;
    std::string name;
  };
}

#endif // TIMESCOPE_H
