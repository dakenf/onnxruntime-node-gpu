cmake_minimum_required(VERSION 3.13)
if(APPLE)
    if(CMAKE_OSX_ARCHITECTURES STREQUAL "")
        message(STATUS "Setting macOS architecture to ${CMAKE_HOST_SYSTEM_PROCESSOR}")
        set(CMAKE_OSX_ARCHITECTURES ${CMAKE_HOST_SYSTEM_PROCESSOR} CACHE STRING "Build architecture for macOS" FORCE)
    endif()
endif()

project(onnxruntime_node_gpu)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if (APPLE)
    add_definitions(-DHAS_COREML)
endif ()
if (WIN32)
    add_definitions(-DHAS_ONNXRUNTIME_CUDA_PROVIDERS)
    add_definitions(-DHAS_DML_PROVIDER_FACTORY)
elseif(UNIX AND NOT APPLE)
#    find_library(ONNXRUNTIME_PROVIDERS_CUDA_LIB NAMES onnxruntime_providers_cuda HINTS ${CUSTOM_LIBRARY_PATHS})
#    if(ONNXRUNTIME_PROVIDERS_CUDA_LIB)
#        message(STATUS "Found ONNX Runtime CUDA providers library: ${ONNXRUNTIME_PROVIDERS_CUDA_LIB}")
#        add_definitions(-DHAS_ONNXRUNTIME_CUDA_PROVIDERS)
#    else()
#        message(STATUS "ONNX Runtime CUDA providers library not found")
#    endif()
endif ()

file(GLOB ORT_NODEJS_BINDING_SOURCE_FILES ${CMAKE_SOURCE_DIR}/src/*.cc)

# Define the Node.js addon target
add_library(${PROJECT_NAME} SHARED ${ORT_NODEJS_BINDING_SOURCE_FILES} ${CMAKE_JS_SRC})
target_include_directories(${PROJECT_NAME} PRIVATE ${CMAKE_JS_INC} ${CMAKE_SOURCE_DIR}/node_modules/node-addon-api)
target_link_directories(${PROJECT_NAME} PRIVATE /usr/local/lib onnxruntime/lib onnxruntime/runtimes/win-x64/native)

include(cmake/FetchOnnxruntime.cmake)

target_link_libraries(${PROJECT_NAME} PRIVATE
        ${CMAKE_JS_LIB}
#        $<$<BOOL:${ONNXRUNTIME_PROVIDERS_CUDA_LIB}>:${ONNXRUNTIME_PROVIDERS_CUDA_LIB}>
#        ${CUDA_LIBRARIES}
        )

# Set the target properties
set_target_properties(${PROJECT_NAME} PROPERTIES PREFIX "" SUFFIX ".node")
target_compile_definitions(${PROJECT_NAME} PRIVATE NAPI_CPP_EXCEPTIONS)
