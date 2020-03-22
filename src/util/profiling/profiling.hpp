#ifndef PROFILING_HPP
#define PROFILING_HPP

#include "util/profiling/timescope.hpp"

#ifdef PROFILE
  #define PROFILE_SCOPE(name) Profiling::TimeScope ts(name)
  #define PROFILE_SECTION_START(m) ts.section_start(m)
  #define PROFILE_SECTION_END() ts.section_end()
#else
  #define PROFILE_SCOPE(name)
  #define PROFILE_SECTION_START(m)
  #define PROFILE_SECTION_END()
#endif

#endif // PROFILING_HPP
