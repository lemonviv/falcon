cmake_minimum_required(VERSION 3.10)
project(falcon)
set(CMAKE_CXX_STANDARD 11)

set(FALCON_INCLUDE_DIR "${PROJECT_SOURCE_DIR}/include/")
include_directories(${FALCON_INCLUDE_DIR};${PROJECT_BINARY_DIR})

SET(PACKAGE_VERSION 1.0.1) # ${VERSION})
SET(VERSION 1.0.1)
SET(FALCON_MAJOR_VERSION 1)
SET(FALCON_MINOR_VERSION 0)
SET(FALCON_PATCH_VERSION 1)

MATH(EXPR FALCON_VERSION "${FALCON_MAJOR_VERSION} * 1000 + ${FALCON_MINOR_VERSION} * 100 + ${FALCON_PATCH_VERSION}")

message(STATUS "FALCON git Version ${VERSION}")
message(STATUS "FALCON Version ${FALCON_VERSION}")

add_subdirectory(src/executor)
enable_testing ()
add_subdirectory(test)