######################################################################################################################
#
# JeVois Smart Embedded Machine Vision Toolkit - Copyright (C) 2016 by Laurent Itti, the University of Southern
# California (USC), and iLab at USC. See http://iLab.usc.edu and http://jevois.org for information about this project.
#
# This file is part of the JeVois Smart Embedded Machine Vision Toolkit.  This program is free software; you can
# redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software
# Foundation, version 2.  This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
# without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public
# License for more details.  You should have received a copy of the GNU General Public License along with this program;
# if not, write to the Free Software Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA.
#
# Contact information: Laurent Itti - 3641 Watt Way, HNB-07A - Los Angeles, BA 90089-2520 - USA.
# Tel: +1 213 740 3527 - itti@pollux.usc.edu - http://iLab.usc.edu - http://jevois.org
######################################################################################################################

## CMake build rules for JeVois Base library and executables

## You may provide the installed JeVois config root as:
#   cmake -DJEVOIS_CONFIG=/jevois/config ..

cmake_minimum_required(VERSION 3.1)

## Set vendor names, our modules will be placed in a directory by that name under /jevois/modules:
set(JEVOIS_VENDOR "SamuelCarbone")
set(JEVOIS_CONFIG "/jevois/config" CACHE STRING "Path to JeVois config to use")

## Include helper functions, config, etc from the JeVois install:
set(CMAKE_MODULE_PATH ${JEVOIS_CONFIG})
include(jevois_config)
include(JeVois)

## Set project name, detects compiler (which has been set by our helper module). Then set some complation flags:
project(firstvision_simplified CXX C ASM)
jevois_project_set_flags()
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

# If your module will use components provided by jevoisbase, uncomment the lines below:
if (JEVOIS_PLATFORM)
  link_directories("/var/lib/jevois-microsd/lib/JeVois") # find libjevoisbase.so for platform
endif (JEVOIS_PLATFORM)
include_directories("/var/lib/jevois-build/usr/include") # find jevoisbase includes

# Added by me to include Eigen/Geometry for the FirstVision code
# Fix path assumptions in the optical flow code for Eigen:
if (JEVOIS_PLATFORM)
  add_definitions(-I${JEVOIS_BUILDROOT_BASE}/host/usr/arm-buildroot-linux-gnueabihf/sysroot/usr/include/eigen3)
  set(BOOST_HEADERS, "/usr/share/jevois-sdk/out/sun8iw5p1/linux/common/buildroot/host/arm-buildroot-linux-gnueabihf/sysroot/usr/include/boost")
  set(PLATFORM_LIB_DIR "/usr/share/jevois-sdk/out/sun8iw5p1/linux/common/buildroot/host/arm-buildroot-linux-gnueabihf/sysroot/usr/lib")
else (JEVOIS_PLATFORM)
  add_definitions(-I/usr/include/eigen3)
  set(BOOST_HEADERS, "usr/include/boost")
endif (JEVOIS_PLATFORM)

# If you also want to use some of the jevoisbase contribs, just add the appropriate include paths here.  You will also
# need to set JEVOIS_SRC_ROOT environment variable in your shell to the parent of the jevoisbase directory, e.g., type:
# export JEVOIS_SRC_ROOT=/home/samuel/Documents
# in your shell.
# include_directories("$ENV{JEVOIS_SRC_ROOT}/jevoisbase/Contrib/darknet-nnpack/include")
# include_directories("$ENV{JEVOIS_SRC_ROOT}/jevoisbase/Contrib/NNPACK/include")
# include_directories("$ENV{JEVOIS_SRC_ROOT}/jevoisbase/Contrib/pthreadpool/include")

# include boost
include_directories(${BOOST_HEADERS})

## Setup our modules that are in src/Modules. First arg: source directory for modules; 2nd arg: target build
## dependencies (i.e., cmake targets which must be built here before we build the modules), usually empty for a single
## module. See the CMakeLists.txt in jevoisbase for an example where we first build a shared library for all shared
## components, and then each module gets this library as a target dependency:
jevois_setup_modules(src/Modules "")

## Add any link libraries for each module. Add 'jevoisbase' here if you want to link against it:
target_link_libraries(FirstVision_Simplified ${JEVOIS_OPENCV_LIBS} opencv_imgproc opencv_core jevoisbase)
target_link_libraries(FirstVision_Simplified jevoisbase)

# link libraries
if(JEVOIS_PLATFORM)
    TARGET_LINK_LIBRARIES( FirstVision_Simplified ${PLATFORM_LIB_DIR}/libboost_system.so ${PLATFORM_LIB_DIR}/libboost_regex.so ${PLATFORM_LIB_DIR}/libboost_thread.so)
    TARGET_LINK_LIBRARIES( FirstVision_Simplified ${PLATFORM_LIB_DIR}/libpthread.so ${PLATFORM_LIB_DIR}/librt.so ${PLATFORM_LIB_DIR}/libtbb.so ${PLATFORM_LIB_DIR}/libdl.so ${PLATFORM_LIB_DIR}/libutil.so ${PLATFORM_LIB_DIR}/libm.so)
else()
    TARGET_LINK_LIBRARIES( FirstVision_Simplified -lboost_system -lboost_regex -lboost_thread -lpthread -lrt -ltbb -ldl -lutil -lm)
endif()

## Install any shared resources (cascade classifiers, neural network weights, etc) in the share/ sub-directory:
install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/share"
  DESTINATION "${JEVOIS_INSTALL_ROOT}" COMPONENT bin)

## Wrapping up:
jevois_project_finalize()  # call this last in your CMakeLists.txt, will un-mount the live microSD if applicable
