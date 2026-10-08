#include "engine/framework/runtime/registry.h"

#include <iostream>
#include <string>
#include <vector>
#include <sstream>

int main() {
    const auto registry = engine::runtime::make_default_registry();
    std::vector<std::string> expected;
    std::istringstream names(AUDIOCPP_EXPECTED_FAMILIES);
    for (std::string name; std::getline(names, name, ',');) expected.push_back(name);
    const auto actual = registry.families();
    if (actual != expected) {
        std::cerr << "unexpected registry families; expected=" << AUDIOCPP_EXPECTED_FAMILIES << " actual=";
        for (const auto & family : actual) std::cerr << family << ",";
        std::cerr << "\n";
        return 1;
    }
    std::cout << "families=" << AUDIOCPP_EXPECTED_FAMILIES << "\n";
    return 0;
}
