#include "timescope.h"
#include "util/profiling/timetree.h"

namespace Profiling {
  using namespace std::chrono;
  static auto time_tree = TimeTree();
  static std::string current_parent;

  TimeScope::TimeScope(const std::string& name)
    : start(steady_clock::now()),
      name(name)
  {
    time_tree.register_element(name);

    if (current_parent == name) {
      return;
    }

    if (current_parent == "") {
      current_parent = name;
      time_tree.register_global_parent(name);
      return;
    }

    if (!time_tree.is_ancestor_of(name, current_parent)) {
      time_tree.register_child(current_parent, name);
    }

    current_parent = name;
  }

  TimeScope::~TimeScope() {
    const auto duration = duration_cast<microseconds>(steady_clock::now() - start).count();
    time_tree.add_time(name, duration);
  }

  void TimeScope::section_start(const std::string& message)
  {
    this->message = message;
    t0 = steady_clock::now();

    time_tree.register_element(message);
    time_tree.register_child(name, message);

    if (current_parent != message) {
      current_parent = message;
    }
  }

  void TimeScope::section_end()
  {
    const auto duration = duration_cast<microseconds>(steady_clock::now() - t0).count();
    time_tree.add_time(message, duration);
  }
}
