#include <fstream>
#include <nlohmann/json.hpp>


nlohmann::json read_json(const std::string file) {
  std::ifstream f(file);
  if (!f.is_open()) {
    std::cout << "failed to open " << file << std::endl;
    exit(EXIT_FAILURE);
  }
  return nlohmann::json::parse(f);
}

std::string to_str(const nlohmann::json val) {
  auto val_str = val.dump();
  if (val_str.front() == val_str.back() == '"') {
    val_str = val_str.substr(1, val_str.length() - 2);
  }
  return val_str;
}
