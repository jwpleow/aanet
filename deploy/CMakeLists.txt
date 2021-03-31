cmake_minimum_required(VERSION 3.0)
project(aanet)

# cmake commands
# cmake -DCMAKE_PREFIX_PATH="$(python -c 'import torch.utils; print(torch.utils.cmake_prefix_path)')" ..
# cmake -DCMAKE_PREFIX_PATH=/home/joel/Downloads/libtorch1.7 ..
# cmake --build . --config Release



# add_definitions(-D_GLIBCXX_USE_CXX11_ABI=0)
# set(CMAKE_CXX_FLAGS "-D_GLIBCXX_USE_CXX11_ABI=0")

# set(CMAKE_FIND_DEBUG_MODE 1)

find_package(Torch REQUIRED)
find_package(OpenCV REQUIRED)

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../nets/deform_conv/dcn_cpp_plugin ${CMAKE_CURRENT_SOURCE_DIR}/../nets/deform_conv/dcn_cpp_plugin/build) # kinda hacky
add_executable(${PROJECT_NAME} main.cpp)
target_link_libraries(${PROJECT_NAME} "${TORCH_LIBRARIES}")
target_link_libraries(${PROJECT_NAME} -Wl,--no-as-needed dcn_v2_cuda_forward_v2)
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})

# message(STATUS "OpenCV include path: ${OpenCV_INCLUDE_DIRS}")
message(STATUS "Torch include path: ${TORCH_LIBRARIES}")

set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 14)
