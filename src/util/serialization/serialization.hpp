#ifndef SERIALIZATION_HPP
#define SERIALIZATION_HPP

#include <iostream>
#include <vector>

template <typename T>
std::istream& operator>>(std::istream& in, std::vector<T>& vec) {
  T t;
  while (in >> t) {
    vec.emplace_back(std::move(t));
  }
  return in;
}

template <typename T>
std::ostream& operator<<(std::ostream& out, const std::vector<T>& vec) {
  for (const auto& el : vec) {
    out << el << std::endl;
  }
  return out;
}

#endif // SERIALIZATION_HPP
