project(YoloV5)

set(${PROJECT_NAME}_SRC
        src/layers.cpp
        src/utils.cpp
        src/yolov5.cpp
    )
    
set(${PROJECT_NAME}_HEADERS
        private_include/YoloV5/layers.hpp
        private_include/YoloV5/utils.hpp
        include/YoloV5/yolov5.hpp
    )
    
set_ide_paths_headers(${PROJECT_NAME}_HEADERS)
set_ide_paths_src(${PROJECT_NAME}_SRC)

add_library(${PROJECT_NAME} SHARED ${${PROJECT_NAME}_SRC} ${${PROJECT_NAME}_HEADERS})
target_include_directories(${PROJECT_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_include_directories(${PROJECT_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/private_include)


target_link_libraries(${PROJECT_NAME} PUBLIC ${TORCH_LIBRARIES})
target_link_libraries(${PROJECT_NAME} PRIVATE ryml)
target_link_libraries(${PROJECT_NAME} PRIVATE WeightsLoading)

set_output(${PROJECT_NAME})
set_debug_names(${PROJECT_NAME})
