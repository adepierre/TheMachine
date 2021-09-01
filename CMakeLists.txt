cmake_minimum_required(VERSION 3.5)

project(TheMachine)

set_property(GLOBAL PROPERTY USE_FOLDERS ON)

# Export symbols from shared libs
set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)

# Enable PIC
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Load/find 3rdparty libraries
add_subdirectory(3rdparty/rapidyaml)
find_package(Torch REQUIRED)
find_package(OpenCV REQUIRED)

include(cmake/utils.cmake)

add_subdirectory(WeightsLoading)
add_subdirectory(YoloV5)
add_subdirectory(TheMachine)




