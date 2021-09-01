project(TheMachine)

set(${PROJECT_NAME}_SRC
        src/main.cpp
        src/TheMachine.cpp
        src/utils.cpp
    )
    
set(${PROJECT_NAME}_HEADERS
        include/TheMachine/TheMachine.hpp
        include/TheMachine/utils.hpp
    )

set_ide_paths_src(${PROJECT_NAME}_SRC)
set_ide_paths_headers(${PROJECT_NAME}_HEADERS)

add_executable(${PROJECT_NAME} ${${PROJECT_NAME}_SRC} ${${PROJECT_NAME}_HEADERS})
target_include_directories(${PROJECT_NAME} PUBLIC include)
target_link_libraries(${PROJECT_NAME} YoloV5)
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})

set_output(${PROJECT_NAME})
set_debug_names(${PROJECT_NAME})

# The following code block is suggested to be used on Windows.
# According to https://github.com/pytorch/pytorch/issues/25457,
# the DLLs need to be copied to avoid memory errors.
if (MSVC)
  file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
  add_custom_command(TARGET ${PROJECT_NAME}
                     POST_BUILD
                     COMMAND ${CMAKE_COMMAND} -E copy_if_different
                     ${TORCH_DLLS}
                     $<TARGET_FILE_DIR:${PROJECT_NAME}>)
endif (MSVC)