cmake_minimum_required(VERSION 3.20)

set(CMAKE_OSX_ARCHITECTURES arm64 CACHE STRING "Target architecture")
set(CMAKE_OSX_DEPLOYMENT_TARGET 13.0 CACHE STRING "Minimum macOS")

project(Feedback LANGUAGES C CXX OBJC OBJCXX)

set(SDK "${CMAKE_SOURCE_DIR}/AfterEffectsSDK/Examples")
set(GEN "${CMAKE_BINARY_DIR}/generated")
file(MAKE_DIRECTORY "${GEN}")

# --- PiPL resource (AE reads this to discover the effect) --------------------
find_program(REZ Rez REQUIRED)
set(PIPL_RSRC "${GEN}/Feedback.rsrc")
add_custom_command(
    OUTPUT "${PIPL_RSRC}"
    COMMAND "${REZ}" -d __MACH__ -useDF -script Roman
            -arch arm64
            -i "${SDK}/Headers" -i "${SDK}/Headers/SP" -i "${SDK}/Resources"
            -o "${PIPL_RSRC}" "${CMAKE_SOURCE_DIR}/src/FeedbackPiPL.r"
    DEPENDS "${CMAKE_SOURCE_DIR}/src/FeedbackPiPL.r"
    COMMENT "Rez: compiling PiPL")

# --- Shaders -----------------------------------------------------------------
set(METALLIB "${GEN}/Feedback.metallib")
add_custom_command(
    OUTPUT "${METALLIB}"
    COMMAND xcrun -sdk macosx metal -std=metal3.0 -O2
            -mmacosx-version-min=${CMAKE_OSX_DEPLOYMENT_TARGET}
            -I "${CMAKE_SOURCE_DIR}/src"
            -o "${METALLIB}" "${CMAKE_SOURCE_DIR}/shaders/Feedback.metal"
    DEPENDS "${CMAKE_SOURCE_DIR}/shaders/Feedback.metal" "${CMAKE_SOURCE_DIR}/src/ShaderTypes.h"
    COMMENT "metal: compiling shaders")

# --- Build stamp ---------------------------------------------------------------
# AE's disk cache outlives the plugin, so frames from a different build mustn't be reused.
# The effect mixes this hash of everything that shapes its pixels into each frame's GUID.
file(GLOB STAMPED_SOURCES CONFIGURE_DEPENDS
    "${CMAKE_SOURCE_DIR}/src/*.mm" "${CMAKE_SOURCE_DIR}/src/*.h" "${CMAKE_SOURCE_DIR}/shaders/*.metal")
set(BUILD_STAMP "${GEN}/BuildStamp.h")
string(REPLACE ";" "|" STAMPED_LIST "${STAMPED_SOURCES}") # a ; would split the command
add_custom_command(
    OUTPUT "${BUILD_STAMP}"
    COMMAND ${CMAKE_COMMAND} "-DSOURCES=${STAMPED_LIST}" "-DOUT=${BUILD_STAMP}"
            -P "${CMAKE_SOURCE_DIR}/cmake/BuildStamp.cmake"
    DEPENDS ${STAMPED_SOURCES} "${CMAKE_SOURCE_DIR}/cmake/BuildStamp.cmake"
    COMMENT "Stamping the build"
    VERBATIM)

# --- Plugin bundle -----------------------------------------------------------
add_library(Feedback MODULE
    src/Feedback.mm
    src/FeedbackCache.mm
    src/FeedbackRenderer.mm
    src/LayerMotion.mm
    "${SDK}/Util/AEFX_SuiteHelper.c"
    "${PIPL_RSRC}"
    "${METALLIB}"
    "${BUILD_STAMP}")

target_include_directories(Feedback PRIVATE
    src
    "${GEN}"
    "${SDK}/Headers"
    "${SDK}/Headers/SP"
    "${SDK}/Util"
    "${SDK}/Resources")

target_compile_features(Feedback PRIVATE cxx_std_17)
set_source_files_properties(src/Feedback.mm src/FeedbackCache.mm src/FeedbackRenderer.mm src/LayerMotion.mm
    PROPERTIES COMPILE_OPTIONS "-fobjc-arc")

option(FEEDBACK_LOG "Log renders to ~/Library/Logs/AEFeedback.log" OFF)
if(FEEDBACK_LOG)
    target_compile_definitions(Feedback PRIVATE FEEDBACK_LOG)
endif()

target_link_libraries(Feedback PRIVATE
    "-framework Foundation"
    "-framework Metal")

set_target_properties(Feedback PROPERTIES
    BUNDLE TRUE
    BUNDLE_EXTENSION plugin
    MACOSX_BUNDLE_INFO_PLIST "${CMAKE_SOURCE_DIR}/resources/Info.plist"
    CXX_VISIBILITY_PRESET hidden
    OBJCXX_VISIBILITY_PRESET hidden
    C_VISIBILITY_PRESET hidden
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")

set(BUNDLE_DIR "$<TARGET_BUNDLE_DIR:Feedback>")
add_custom_command(TARGET Feedback POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory "${BUNDLE_DIR}/Contents/Resources"
    COMMAND ${CMAKE_COMMAND} -E copy "${PIPL_RSRC}" "${BUNDLE_DIR}/Contents/Resources/Feedback.rsrc"
    COMMAND ${CMAKE_COMMAND} -E copy "${METALLIB}" "${BUNDLE_DIR}/Contents/Resources/Feedback.metallib"
    COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_SOURCE_DIR}/resources/PkgInfo" "${BUNDLE_DIR}/Contents/PkgInfo"
    COMMAND codesign --force --sign - "${BUNDLE_DIR}"
    # AE caches each plugin's PiPL and only rescans when the bundle's own date changes,
    # which replacing the files inside it doesn't do.
    COMMAND ${CMAKE_COMMAND} -E touch "${BUNDLE_DIR}"
    COMMENT "Assembling and ad-hoc signing Feedback.plugin")
