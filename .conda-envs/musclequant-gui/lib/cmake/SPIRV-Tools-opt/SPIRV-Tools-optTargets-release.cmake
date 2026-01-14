#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "SPIRV-Tools-opt" for configuration "Release"
set_property(TARGET SPIRV-Tools-opt APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(SPIRV-Tools-opt PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libSPIRV-Tools-opt.dylib"
  IMPORTED_SONAME_RELEASE "@rpath/libSPIRV-Tools-opt.dylib"
  )

list(APPEND _cmake_import_check_targets SPIRV-Tools-opt )
list(APPEND _cmake_import_check_files_for_SPIRV-Tools-opt "${_IMPORT_PREFIX}/lib/libSPIRV-Tools-opt.dylib" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
