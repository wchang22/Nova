#ifndef EXCEPTION_HPP
#define EXCEPTION_HPP

#include <stdexcept>
#include <string>

#define GENERATE_EXCEPTION_HEADER(name)     \
  struct name : public std::runtime_error { \
    name(const std::string& msg);           \
  };

#define GENERATE_EXCEPTION_IMPL(name) \
  name::name(const std::string& msg) : std::runtime_error(#name + std::string(": ") + msg) {}

GENERATE_EXCEPTION_HEADER(FileException)
GENERATE_EXCEPTION_HEADER(LoggingException)
GENERATE_EXCEPTION_HEADER(KernelException)
GENERATE_EXCEPTION_HEADER(ModelException)
GENERATE_EXCEPTION_HEADER(SkyException)
GENERATE_EXCEPTION_HEADER(ImageException)
GENERATE_EXCEPTION_HEADER(TriangleException)
GENERATE_EXCEPTION_HEADER(AcceleratorException)
GENERATE_EXCEPTION_HEADER(WindowException)
GENERATE_EXCEPTION_HEADER(DenoiseException)

#endif // EXCEPTION_HPP
