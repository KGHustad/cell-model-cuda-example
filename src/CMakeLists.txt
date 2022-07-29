cmake_minimum_required(VERSION 3.9...3.22 FATAL_ERROR)

project(cell-model-cuda-example LANGUAGES CUDA C CXX)

set(CUDA_ARCH_MINIMUM 35)
set(CUDA_ARCH_FLAGS "")
set(CUDA_GENCODE_FLAGS "")

set(CUDA_ARCHS_DESC "Semi-colon-separated list of CUDA compute capabilities to target (e.g. \"70;80\" for 7.0 and 8.0)")
set(CUDA_ARCHS "" CACHE STRING "${CUDA_ARCHS_DESC}")
#mark_as_advanced(CLEAR CUDA_ARCHS)
if (CUDA_ARCHS STREQUAL "")
    message(STATUS "Detecting CUDA architectures")
    # detect CUDA architecture for each GPU in the current systen
    set(SOURCEFILE ${PROJECT_SOURCE_DIR}/cmake/detect_compute_capabilities.cu)
    try_run(DETECT_COMPUTE_CAPABILITIES_RUN_EXITCODE DETECT_COMPUTE_CAPABILITIES_COMPILE_SUCCESS
        ${PROJECT_BINARY_DIR} ${SOURCEFILE}
        RUN_OUTPUT_VARIABLE DETECTED_CUDA_ARCHS
        )
    #message(NOTICE "DETECT_COMPUTE_CAPABILITIES_COMPILE_SUCCESS: ${DETECT_COMPUTE_CAPABILITIES_COMPILE_SUCCESS}")
    #message(NOTICE "DETECT_COMPUTE_CAPABILITIES_RUN_EXITCODE: ${DETECT_COMPUTE_CAPABILITIES_RUN_EXITCODE}")
    #message(NOTICE "DETECTED_CUDA_ARCHS: ${DETECTED_CUDA_ARCHS}")
    set(CUDA_ARCHS "${DETECTED_CUDA_ARCHS}")
endif()

list(SORT CUDA_ARCHS)
list(REMOVE_DUPLICATES CUDA_ARCHS)
list(LENGTH CUDA_ARCHS NUM_CUDA_ARCHS)

# update the value seen by the user
set(CUDA_ARCHS "${CUDA_ARCHS}" CACHE STRING "${CUDA_ARCHS_DESC}" FORCE)

if(NUM_CUDA_ARCHS)
    list(GET CUDA_ARCHS 0 CUDA_ARCH_MINIMUM_DETECTED)
    if(CUDA_ARCH_MINIMUM_DETECTED LESS CUDA_ARCH_MINIMUM)
        message(WARNING "The CUDA backend requires CUDA arch ${CUDA_ARCH_MINIMUM} or newer, but a device with CUDA arch ${CUDA_ARCH_MINIMUM} was detected in the system.")
    else()
        # Use the lowest CUDA arch as the minimum
        set(CUDA_ARCH_MINIMUM "${CUDA_ARCH_MINIMUM_DETECTED}")
    endif()
endif()

set(CUDA_ARCH_FLAGS "--gpu-architecture=sm_${CUDA_ARCH_MINIMUM}")
if(${CMAKE_VERSION} VERSION_LESS "3.18.0")
    foreach(CUDA_ARCH IN LISTS CUDA_ARCHS)
        set(CUDA_GENCODE_FLAGS "${CUDA_GENCODE_FLAGS} -gencode arch=compute_${CUDA_ARCH},code=sm_${CUDA_ARCH}")
    endforeach(CUDA_ARCH)
else()
    # CMake >= 3.18 wants to set the flags itself
    set(CMAKE_CUDA_ARCHITECTURES ${CUDA_ARCHS})
endif()

add_executable(TP06
    cli.c
    TP06.cu
)
