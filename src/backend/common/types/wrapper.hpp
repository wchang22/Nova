#ifndef COMMON_WRAPPER_HPP
#define COMMON_WRAPPER_HPP

#include <utility>

namespace nova {

template <typename T>
class Wrapper {
public:
  Wrapper() = default;

  template <typename... Args>
  Wrapper(Args&&... args) : value(std::forward<Args>(args)...) {}

  Wrapper(const Wrapper& other) : value(other.value) {}
  Wrapper(Wrapper& other) : value(other.value) {}

  Wrapper& operator=(const Wrapper& other) {
    value = other.value;
    return *this;
  }
  Wrapper& operator=(Wrapper& other) {
    value = other.value;
    return *this;
  }

  Wrapper(Wrapper&& other) : value(std::move(other.value)) {}
  Wrapper& operator=(Wrapper&& other) {
    std::swap(value, other.value);
    return *this;
  }

  ~Wrapper() = default;

  const T& data() const { return value; }
  T& data() { return value; }

private:
  T value;
};

}

#endif // COMMON_WRAPPER_HPP
