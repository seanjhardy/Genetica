cmake_minimum_required(VERSION 3.8)
project(Genetica LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 26)
set(CMAKE_CUDA_STANDARD 26)

file(GLOB_RECURSE SRCS "src/*.cpp")
file(GLOB_RECURSE CUDA_SRCS "src/*.cu")
file(GLOB_RECURSE HEADERS "./*.hpp")

set_source_files_properties(main.cpp ${SRCS} PROPERTIES LANGUAGE CXX)
set_source_files_properties(${CUDA_SRCS} PROPERTIES LANGUAGE CUDA)

# Add the executable
add_executable(Genetica main.cpp ${SRCS} ${CUDA_SRCS} ${HEADERS})

include_directories(./include
        "c:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.5/include")

target_link_libraries(Genetica ${CMAKE_SOURCE_DIR}/library/sfml-window-d.lib)
target_link_libraries(Genetica ${CMAKE_SOURCE_DIR}/library/sfml-graphics-d.lib)
target_link_libraries(Genetica ${CMAKE_SOURCE_DIR}/library/sfml-audio-d.lib)
target_link_libraries(Genetica ${CMAKE_SOURCE_DIR}/library/sfml-network-d.lib)
target_link_libraries(Genetica ${CMAKE_SOURCE_DIR}/library/sfml-main-d.lib)
target_link_libraries(Genetica ${CMAKE_SOURCE_DIR}/library/sfml-system-d.lib)

set_target_properties(Genetica PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(Genetica PROPERTIES CUDA_ARCHITECTURES "52")

file(COPY assets DESTINATION ${CMAKE_BINARY_DIR})

# Copy dlls to build
if(WIN32)
    file(GLOB BINARY_DEP_DLLS "c:/tools/SFML-wvc/bin/*.dll")
    file(COPY ${BINARY_DEP_DLLS} DESTINATION ${CMAKE_BINARY_DIR})

    file(GLOB MINGW_DEP_DLLS "C:/mingw64/bin/*.dll")
    file(COPY ${MINGW_DEP_DLLS} DESTINATION ${CMAKE_BINARY_DIR})
endif()

message("${CMAKE_BINARY_DIR} ${CMAKE_CURRENT_BINARY_DIR}")
