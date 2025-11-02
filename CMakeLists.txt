cmake_minimum_required(VERSION 3.8)
project(Genetica LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 26)

file(GLOB_RECURSE SRCS "src/*.cpp")
file(GLOB_RECURSE HEADERS "./*.hpp" "src/*.tpp")

# Add Objective-C++ files on macOS
if(APPLE)
    file(GLOB_RECURSE OBJCPP_SRCS "src/*.mm")
    set(SRCS ${SRCS} ${OBJCPP_SRCS})
endif()

file(GLOB_RECURSE XMLS "src/*.xml")
file(GLOB_RECURSE CSSS "src/*.css")

set_source_files_properties(main.cpp ${SRCS} PROPERTIES LANGUAGE CXX)

# Add the executable
if(APPLE)
    add_executable(Genetica MACOSX_BUNDLE main.cpp ${SRCS})
    set(MACOSX_BUNDLE_ICON_FILE Genetica.icns)
    set(MACOSX_BUNDLE_BUNDLE_NAME "Genetica")
    set(MACOSX_BUNDLE_GUI_IDENTIFIER "com.genetica.app")
    set(MACOSX_BUNDLE_SHORT_VERSION_STRING "1.0")
    set(MACOSX_BUNDLE_BUNDLE_VERSION "1.0.0")
    
    # Copy icon to Resources folder
    set_source_files_properties(${CMAKE_SOURCE_DIR}/Genetica.icns PROPERTIES
        MACOSX_PACKAGE_LOCATION "Resources")
    target_sources(Genetica PRIVATE ${CMAKE_SOURCE_DIR}/Genetica.icns)
else()
    add_executable(Genetica main.cpp ${SRCS})
endif()

file(GLOB_RECURSE INCLUDE_LIBS "${CMAKE_SOURCE_DIR}/libs/*/include")

include_directories(./include
        #${CMAKE_SOURCE_DIR}/libs/SFML/include
        ${CMAKE_SOURCE_DIR}/libs/GLEW/include
        )

# Add OpenCL example
add_executable(OpenCLExample opencl_example.cpp)
target_include_directories(OpenCLExample PRIVATE ./include)

# Platform-specific library linking
if(WIN32)
    file(GLOB_RECURSE LIBS "${CMAKE_SOURCE_DIR}/libs/*.lib")
    target_link_libraries(Genetica ${LIBS})
elseif(APPLE)
    # Use system SFML 2.5 on macOS
    set(CMAKE_PREFIX_PATH "/opt/homebrew/opt/sfml@2")
    find_package(SFML 2.5 COMPONENTS graphics window audio network system REQUIRED)
    target_link_libraries(Genetica sfml-graphics sfml-window sfml-audio sfml-network sfml-system)
    
    # Link against Cocoa and OpenGL frameworks for native window access
    find_library(COCOA_LIBRARY Cocoa)
    find_library(OPENGL_LIBRARY OpenGL)
    target_link_libraries(Genetica ${COCOA_LIBRARY} ${OPENGL_LIBRARY})
else()
    # For Linux
    find_package(SFML 2.5 COMPONENTS graphics window audio network system REQUIRED)
    target_link_libraries(Genetica sfml-graphics sfml-window sfml-audio sfml-network sfml-system)
endif()


# Copy assets
file(REMOVE_RECURSE "${CMAKE_BINARY_DIR}/assets/")
file(COPY assets DESTINATION ${CMAKE_BINARY_DIR})
file(COPY ${XMLS} DESTINATION "${CMAKE_BINARY_DIR}/assets/components")
file(COPY ${CSSS} DESTINATION "${CMAKE_BINARY_DIR}/assets/styles")

# For macOS app bundle, also copy assets into the bundle
if(APPLE)
    add_custom_command(TARGET Genetica POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${CMAKE_SOURCE_DIR}/assets
        $<TARGET_FILE_DIR:Genetica>/../Resources/assets
    )
    add_custom_command(TARGET Genetica POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${CMAKE_BINARY_DIR}/assets/components
        $<TARGET_FILE_DIR:Genetica>/../Resources/assets/components
    )
    add_custom_command(TARGET Genetica POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${CMAKE_BINARY_DIR}/assets/styles
        $<TARGET_FILE_DIR:Genetica>/../Resources/assets/styles
    )
endif()

# Copy dlls to build
if(WIN32)
    file(GLOB_RECURSE BINARY_DEP_DLLS "${CMAKE_SOURCE_DIR}/libs/*.dll")
    file(COPY ${BINARY_DEP_DLLS} DESTINATION ${CMAKE_BINARY_DIR})

    file(GLOB MINGW_DEP_DLLS "C:/mingw64/bin/*.dll")
    file(COPY ${MINGW_DEP_DLLS} DESTINATION ${CMAKE_BINARY_DIR})
endif()

# OpenCL for main project and example
if(APPLE)
    find_library(OpenCL_LIBRARY OpenCL)
    target_link_libraries(Genetica ${OpenCL_LIBRARY})
    target_link_libraries(OpenCLExample ${OpenCL_LIBRARY})
else()
    find_package(OpenCL REQUIRED)
    target_include_directories(Genetica ${OpenCL_INCLUDE_DIRS})
    target_link_libraries(Genetica ${OpenCL_LIBRARIES})
    target_include_directories(OpenCLExample PRIVATE ${OpenCL_INCLUDE_DIRS})
    target_link_libraries(OpenCLExample ${OpenCL_LIBRARIES})
endif()


# Copy kernel file to build directory
configure_file(opencl_example_kernel.cl ${CMAKE_BINARY_DIR}/opencl_example_kernel.cl COPYONLY)

message("${CMAKE_BINARY_DIR} ${CMAKE_CURRENT_BINARY_DIR}")
