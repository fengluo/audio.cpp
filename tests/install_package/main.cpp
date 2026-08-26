#include "engine/framework/runtime/registry.h"

#include <iostream>
#include <string>
#include <vector>

int main() {
    const auto registry = engine::runtime::make_default_registry();
    const std::vector<std::string> expected = {
        "marblenet_vad",
        "silero_vad",
    };
    const auto actual = registry.families();
    if (actual != expected) {
        std::cerr << "unexpected registry families\n";
        return 1;
    }
    std::cout << "families=marblenet_vad,silero_vad\n";
    return 0;
}
