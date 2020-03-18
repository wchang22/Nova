#include "timetree.hpp"
#include "util/logging/logging.hpp"

#include <sstream>
#include <numeric>
#include <iomanip>

TimeTree::~TimeTree()
{
  logger_t logger = Logging::get_logger();
  logger << "-----------------------------------------------------" << std::endl;
  logger << "                       Timings                       " << std::endl;
  logger << "-----------------------------------------------------" << std::endl;
  logger << print_tree();
}

void TimeTree::register_element(const std::string& element)
{
  auto [it, success] = time_map.try_emplace(element, std::vector<long>());

  if (!success) {
    return;
  }

  hierarchy.try_emplace(element, std::vector<std::string>());
}

void TimeTree::register_child(const std::string& parent, const std::string& child)
{
  auto [it, success] = hierarchy_search[parent].emplace(child);

  if (!success) {
    return;
  }

  time_map.try_emplace(child, std::vector<long>());
  hierarchy[parent].emplace_back(child);
}

void TimeTree::add_time(const std::string& name, long time)
{
  time_map[name].emplace_back(time);
}

bool TimeTree::is_ancestor_of(const std::string& ancestor, const std::string& child)
{
  const auto& children = hierarchy_search[ancestor];
  if (children.find(child) != children.end()) {
    return true;
  }

  for (const auto& parent : children) {
    if (is_ancestor_of(parent, child)) {
      return true;
    }
  }

  return false;
}

void TimeTree::register_global_parent(const std::string& parent)
{
  register_element(parent);
  global_parent = parent;
}

double get_average_time(const std::vector<long>& times) {
  return std::accumulate(times.begin(), times.end(), 0.0, [&times] (auto a, auto b) {
    return (a * times.size() + b) / times.size();
  });
}

void TimeTree::print_average(std::ostream& stream, const std::string& name, int padding) {
  for (int i = 0; i < padding; i++) {
    stream << " ";
  }

  double time = get_average_time(time_map.find(name)->second);
  const char* unit = "us";

  if (time >= 1e6) {
    time /= 1e6;
    unit = "s";
  } else if (time >= 1e3) {
    time /= 1e3;
    unit = "ms";
  }

  stream << name;
  stream << std::right << std::setw(50 - padding - static_cast<int>(name.length()))
         << time << " " << unit << std::endl;
}

void TimeTree::print_element(std::ostream& stream, const std::string& name, int padding)
{
  print_average(stream, name, padding);
  auto children = hierarchy.find(name);

  if (children == hierarchy.end()) {
    return;
  }

  for (const auto& child : children->second) {
   print_element(stream, child, padding + 2);
  }
}

std::string TimeTree::print_tree()
{
  if (global_parent.empty()) {
    return "";
  }

  std::stringstream ss;
  print_element(ss, global_parent, 0);

  return ss.str();
}
