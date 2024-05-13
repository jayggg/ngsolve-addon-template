###############################################################################
# This file was taken from https://github.com/NGSolve/ngsolve-addon-template
# Make sure to check for updates regularly.
# Don't change anything here this line (unless you know what you are doing!)
###############################################################################
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Find NGSolve and Netgen using python
if(CMAKE_VERSION VERSION_LESS "3.18")
  find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
else()
  find_package(Python3 REQUIRED COMPONENTS Interpreter Development.Module)
endif()

set(Netgen_DIR "" CACHE PATH "Path to directory containing NetgenConfig.cmake")
set(NGSolve_DIR "" CACHE PATH "Path to directory containing NGSolveConfig.cmake")

execute_process(COMMAND ${Python3_EXECUTABLE} -m netgen.config OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE Netgen_DIR)
execute_process(COMMAND ${Python3_EXECUTABLE} -m ngsolve.config OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE NGSolve_DIR)

find_package(NGSolve CONFIG REQUIRED)

# Create the module
add_library(${module_name} SHARED ${source_files})
target_link_libraries(${module_name} PUBLIC ngsolve Python3::Module)
set_target_properties(${module_name} PROPERTIES PREFIX "" CXX_STANDARD 17)

# Python does not recognize .dll (Windows) and .dylib (MacOS) file endings as modules
if(WIN32)
  set_target_properties(${module_name} PROPERTIES SUFFIX ".pyd" )
else(WIN32)
  set_target_properties(${module_name} PROPERTIES SUFFIX ".so")
endif(WIN32)

execute_process(COMMAND ${Python3_EXECUTABLE} -c "import sys,sysconfig,os.path; print(os.path.relpath(sysconfig.get_path('platlib'), sys.prefix))"
  OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE python3_library_dir
)

if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
  # Set install prefix to user-base if a user site is available, sys.prefix otherwise
  execute_process(COMMAND ${Python3_EXECUTABLE} -c "import sys; print(sys.prefix)"
    OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE install_prefix
  )
  execute_process(COMMAND ${Python3_EXECUTABLE} -m site --user-base
    OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE user_base RESULT_VARIABLE ret
  )
  if (ret EQUAL 0)
    set(install_prefix ${user_base})
  endif()
  set(CMAKE_INSTALL_PREFIX ${install_prefix} CACHE PATH "Install dir" FORCE)
  set(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT OFF)
endif(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)

message(STATUS "Install prefix: ${CMAKE_INSTALL_PREFIX}")
message(STATUS "Module install dir: ${python3_library_dir}")
install(TARGETS ${module_name} DESTINATION ${python3_library_dir})
