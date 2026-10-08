if (NOT DEFINED AUDIOCPP_SOURCE_DIR OR NOT DEFINED AUDIOCPP_BINARY_DIR)
    message(FATAL_ERROR "AUDIOCPP_SOURCE_DIR and AUDIOCPP_BINARY_DIR are required")
endif()

set(prefix "${AUDIOCPP_BINARY_DIR}/install-package-prefix")
set(consumer_binary_dir "${AUDIOCPP_BINARY_DIR}/install-package-consumer")
file(REMOVE_RECURSE "${prefix}" "${consumer_binary_dir}")

execute_process(
    COMMAND "${CMAKE_COMMAND}" --install "${AUDIOCPP_BINARY_DIR}" --prefix "${prefix}"
    RESULT_VARIABLE install_result
)
if (NOT install_result EQUAL 0)
    message(FATAL_ERROR "AudioCpp package installation failed: ${install_result}")
endif()

execute_process(
    COMMAND "${CMAKE_COMMAND}"
        -S "${AUDIOCPP_SOURCE_DIR}/tests/install_package"
        -B "${consumer_binary_dir}"
        -DCMAKE_PREFIX_PATH=${prefix}
    RESULT_VARIABLE configure_result
)
if (NOT configure_result EQUAL 0)
    message(FATAL_ERROR "AudioCpp package consumer configure failed: ${configure_result}")
endif()

execute_process(
    COMMAND "${CMAKE_COMMAND}" --build "${consumer_binary_dir}" --parallel 4
    RESULT_VARIABLE build_result
)
if (NOT build_result EQUAL 0)
    message(FATAL_ERROR "AudioCpp package consumer build failed: ${build_result}")
endif()

execute_process(
    COMMAND "${consumer_binary_dir}/audiocpp_install_package_fixture"
    RESULT_VARIABLE fixture_result
    OUTPUT_VARIABLE fixture_output
)
if (NOT fixture_result EQUAL 0)
    message(FATAL_ERROR "AudioCpp package consumer failed: ${fixture_result}")
endif()
if (NOT fixture_output MATCHES "families=")
    message(FATAL_ERROR "AudioCpp package consumer returned unexpected output: ${fixture_output}")
endif()

file(READ "${prefix}/lib/cmake/AudioCpp/AudioCppTargets.cmake" target_contents)
string(FIND "${target_contents}" "${AUDIOCPP_SOURCE_DIR}" source_path_index)
string(FIND "${target_contents}" "${AUDIOCPP_BINARY_DIR}" build_path_index)
if (NOT source_path_index EQUAL -1 OR NOT build_path_index EQUAL -1)
    message(FATAL_ERROR "AudioCpp package exports a source or build directory")
endif()
