cmake_minimum_required(VERSION 3.20)
project(slang_hello CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(SLANG_SDK "/Users/j/Downloads/slang-2025.15.1" CACHE PATH "slang install directory")

add_executable(slang_hello src/main.cpp)

target_include_directories(slang_hello PRIVATE
    "${SLANG_SDK}/include"
)

target_link_libraries(slang_hello PRIVATE
    "${SLANG_SDK}/lib/libslang.dylib"
)

set_target_properties(slang_hello PROPERTIES
    BUILD_RPATH "${SLANG_SDK}/lib"
)

set_target_properties(slang_hello PROPERTIES
    INSTALL_RPATH "@loader_path/../lib"
)
