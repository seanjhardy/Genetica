cmake_minimum_required(VERSION 3.8)
project(Genetica LANGUAGES CXX CUDA)

set(CMAKE_CUDA_ARCHITECTURES "native")
set(CMAKE_CXX_STANDARD 26)
set(CMAKE_CUDA_STANDARD 26)

file(GLOB_RECURSE SRCS "src/*.cpp")
file(GLOB_RECURSE CUDA_SRCS "src/*.cu")
file(GLOB_RECURSE HEADERS "./*.hpp" "src/*.tpp")

file(GLOB_RECURSE XMLS "src/*.xml")
file(GLOB_RECURSE CSSS "src/*.css")

set_source_files_properties(main.cpp ${SRCS} PROPERTIES LANGUAGE CXX)
set_source_files_properties(${CUDA_SRCS} PROPERTIES LANGUAGE CUDA)

# Add the executable
add_executable(Genetica main.cpp ${SRCS} ${CUDA_SRCS})

file(GLOB_RECURSE INCLUDE_LIBS "${CMAKE_SOURCE_DIR}/libs/*/include")

include_directories(./include
        ${CMAKE_SOURCE_DIR}/libs/SFML/include
        ${CMAKE_SOURCE_DIR}/libs/GLEW/include
        "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.6/include")

file(GLOB_RECURSE LIBS "${CMAKE_SOURCE_DIR}/libs/*.lib")
target_link_libraries(Genetica ${LIBS})

set_target_properties(Genetica PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(Genetica PROPERTIES CUDA_ARCHITECTURES "52")

file(REMOVE_RECURSE "${CMAKE_BINARY_DIR}/assets/")
file(COPY assets DESTINATION ${CMAKE_BINARY_DIR})
file(COPY ${XMLS} DESTINATION "${CMAKE_BINARY_DIR}/assets/components")
file(COPY ${CSSS} DESTINATION "${CMAKE_BINARY_DIR}/assets/styles")

# Copy dlls to build
if(WIN32)
    file(GLOB_RECURSE BINARY_DEP_DLLS "${CMAKE_SOURCE_DIR}/libs/*.dll")
    file(COPY ${BINARY_DEP_DLLS} DESTINATION ${CMAKE_BINARY_DIR})

    file(GLOB MINGW_DEP_DLLS "C:/mingw64/bin/*.dll")
    file(COPY ${MINGW_DEP_DLLS} DESTINATION ${CMAKE_BINARY_DIR})
endif()

message("${CMAKE_BINARY_DIR} ${CMAKE_CURRENT_BINARY_DIR}")
