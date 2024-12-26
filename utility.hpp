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
