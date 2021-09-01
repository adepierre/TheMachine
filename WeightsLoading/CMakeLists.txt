project(WeightsLoading)

set(${PROJECT_NAME}_SRC
        src/weights_loader.cpp
    )
    
set(${PROJECT_NAME}_HEADERS
        include/WeightsLoading/weights_loader.hpp
    )
    
set_ide_paths_headers(${PROJECT_NAME}_HEADERS)
set_ide_paths_src(${PROJECT_NAME}_SRC)

add_library(${PROJECT_NAME} SHARED ${${PROJECT_NAME}_SRC} ${${PROJECT_NAME}_HEADERS})
target_include_directories(${PROJECT_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

set_output(${PROJECT_NAME})
set_debug_names(${PROJECT_NAME})