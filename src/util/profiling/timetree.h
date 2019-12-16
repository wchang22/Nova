#ifndef TIMETREE_H
#define TIMETREE_H

#include <unordered_map>
#include <unordered_set>
#include <vector>

class TimeTree
{
public:
  ~TimeTree();

  void register_element(const std::string& element);
  void register_child(const std::string& parent, const std::string& child);
  void add_time(const std::string& name, long time);
  bool is_ancestor_of(const std::string& ancestor, const std::string& child);
  void register_global_parent(const std::string& parent);
  std::string print_tree();

private:
  void print_average(std::ostream& stream, const std::string& name, int padding);
  void print_element(std::ostream& stream, const std::string& name, int padding);

  std::unordered_map<std::string, std::vector<long>> time_map;
  std::unordered_map<std::string, std::vector<std::string>> hierarchy;
  std::unordered_map<std::string, std::unordered_set<std::string>> hierarchy_search;
  std::string global_parent = "";
};

#endif // TIMETREE_H
