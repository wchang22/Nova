#include <stack>

#include "timescope.hpp"
#include "util/profiling/timetree.hpp"

namespace Profiling {
using namespace std::chrono;
static auto time_tree = TimeTree();
static std::stack<std::string> parent_stack;

TimeScope::TimeScope(const std::string& name) : start(steady_clock::now()), name(name) {
  time_tree.register_element(name);

  if (parent_stack.empty()) {
    parent_stack.push(name);
    time_tree.register_global_parent(name);
    return;
  }

  if (parent_stack.top() == name) {
    return;
  }

  if (!time_tree.is_ancestor_of(name, parent_stack.top())) {
    time_tree.register_child(parent_stack.top(), name);
  }

  parent_stack.push(name);
}

TimeScope::~TimeScope() {
  const auto duration = duration_cast<microseconds>(steady_clock::now() - start).count();
  time_tree.add_time(name, duration);
  parent_stack.pop();
}

void TimeScope::section_start(const std::string& message) {
  this->message = message;
  t0 = steady_clock::now();

  time_tree.register_element(message);

  if (parent_stack.top() != message) {
    time_tree.register_child(name, message);
    parent_stack.push(message);
  }
}

void TimeScope::section_end() {
  const auto duration = duration_cast<microseconds>(steady_clock::now() - t0).count();
  time_tree.add_time(message, duration);
  parent_stack.pop();
}
}
