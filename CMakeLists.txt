cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(TheMachine)

add_subdirectory(3rdparty/rapidyaml)

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

set(${CMAKE_PROJECT_NAME}_SRC
        src/main.cpp
        src/layers.cpp
        src/utils.cpp
        src/yolov5.cpp
    )
    
set(${CMAKE_PROJECT_NAME}_HEADERS
        include/TheMachine/layers.hpp
        include/TheMachine/utils.hpp
        include/TheMachine/yolov5.hpp
    )

# To have a nice files structure in Visual Studio
if(MSVC)
    foreach(source IN LISTS ${CMAKE_PROJECT_NAME}_HEADERS)
        get_filename_component(source_path_header "${source}" PATH)
        string(REPLACE "include/${CMAKE_PROJECT_NAME}" "Headers/" source_path_header "${source_path_header}")
        string(REPLACE "/" "\\" source_path_msvc "${source_path_header}")
        source_group("${source_path_msvc}" FILES "${source}")
    endforeach()
    
    foreach(source IN LISTS ${CMAKE_PROJECT_NAME}_SRC)
        get_filename_component(source_path "${source}" PATH)
        string(REPLACE "src" "Sources" source_path "${source_path}")
        string(REPLACE "/" "\\" source_path_msvc "${source_path}")
        source_group("${source_path_msvc}" FILES "${source}")
    endforeach()
endif()


add_executable(${CMAKE_PROJECT_NAME} ${${CMAKE_PROJECT_NAME}_HEADERS} ${${CMAKE_PROJECT_NAME}_SRC})
target_include_directories(${CMAKE_PROJECT_NAME} PUBLIC include)
target_link_libraries(${CMAKE_PROJECT_NAME} "${TORCH_LIBRARIES}")
target_link_libraries(${CMAKE_PROJECT_NAME} ryml)
set_property(TARGET ${CMAKE_PROJECT_NAME} PROPERTY CXX_STANDARD 14)


# The following code block is suggested to be used on Windows.
# According to https://github.com/pytorch/pytorch/issues/25457,
# the DLLs need to be copied to avoid memory errors.
if (MSVC)
  file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
  add_custom_command(TARGET ${CMAKE_PROJECT_NAME}
                     POST_BUILD
                     COMMAND ${CMAKE_COMMAND} -E copy_if_different
                     ${TORCH_DLLS}
                     $<TARGET_FILE_DIR:${CMAKE_PROJECT_NAME}>)
endif (MSVC)