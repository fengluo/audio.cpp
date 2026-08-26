cmake_minimum_required(VERSION 3.20)

project(AudioCppInstallPackageFixture LANGUAGES CXX)

find_package(AudioCpp CONFIG REQUIRED)

add_executable(audiocpp_install_package_fixture main.cpp)
target_link_libraries(audiocpp_install_package_fixture PRIVATE AudioCpp::engine_runtime)
