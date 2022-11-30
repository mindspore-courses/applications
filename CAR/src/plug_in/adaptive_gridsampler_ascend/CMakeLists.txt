# Copyright (c) Huawei Technologies Co., Ltd. 2022. All rights reserved.

cmake_minimum_required(VERSION 3.5.1)
# add_definitions(-D_GLIBCXX_USE_CXX11_ABI=0)
message(STATUS "CMAKE_CURRENT_BINARY_DIR=${CMAKE_CURRENT_BINARY_DIR}")
message(STATUS "CMAKE_CURRENT_SOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}")

project(opp)

if(EXISTS ${CMAKE_CURRENT_BINARY_DIR}/cmake/config.cmake)
    include(${CMAKE_CURRENT_BINARY_DIR}/cmake/config.cmake)
else()
    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/cmake/config.cmake)
        include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/config.cmake)
    endif()
endif()

execute_process(COMMAND rm -rf ./makepkg)

message(STATUS "OP_COMMON_INC=${OP_COMMON_INC}")
# add sub compile directory
if(EXISTS "${CMAKE_SOURCE_DIR}/cpukernel")
    if (NOT "x$ENV{COMPILE_KERNEL_TARGET}" STREQUAL "xtbe")
        message(STATUS "compile kernel target is $ENV{COMPILE_KERNEL_TARGET}")
        add_subdirectory(cpukernel)
    endif()
endif()

message(STATUS "operation system is ${CMAKE_HOST_SYSTEM_NAME}")

set(ALL_MODULES ${OP_PROTO_TARGET})
if(EXISTS "${CMAKE_SOURCE_DIR}/cpukernel/CMakeLists.txt")
    if (NOT "x$ENV{COMPILE_KERNEL_TARGET}" STREQUAL "xtbe")
        message(STATUS "kernel target and config json: $ENV{COMPILE_KERNEL_TARGET}")
        set(ALL_MODULES ${ALL_MODULES} ${AICPU_KERNEL_TARGET})
        set(ALL_MODULES ${ALL_MODULES} ${AICPU_CONFIG_JSON_TARGET})
    endif()
endif()
message(STATUS "AICPU_CONFIG_INFO_FLAG=${AICPU_CONFIG_INFO_FLAG}")
