#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "sleef::sleefscalar" for configuration "Release"
set_property(TARGET sleef::sleefscalar APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(sleef::sleefscalar PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libsleefscalar.3.9.0.dylib"
  IMPORTED_SONAME_RELEASE "@rpath/libsleefscalar.3.dylib"
  )

list(APPEND _cmake_import_check_targets sleef::sleefscalar )
list(APPEND _cmake_import_check_files_for_sleef::sleefscalar "${_IMPORT_PREFIX}/lib/libsleefscalar.3.9.0.dylib" )

# Import target "sleef::sleef" for configuration "Release"
set_property(TARGET sleef::sleef APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(sleef::sleef PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libsleef.3.9.0.dylib"
  IMPORTED_SONAME_RELEASE "@rpath/libsleef.3.dylib"
  )

list(APPEND _cmake_import_check_targets sleef::sleef )
list(APPEND _cmake_import_check_files_for_sleef::sleef "${_IMPORT_PREFIX}/lib/libsleef.3.9.0.dylib" )

# Import target "sleef::sleefquad" for configuration "Release"
set_property(TARGET sleef::sleefquad APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(sleef::sleefquad PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libsleefquad.3.9.0.dylib"
  IMPORTED_SONAME_RELEASE "@rpath/libsleefquad.3.dylib"
  )

list(APPEND _cmake_import_check_targets sleef::sleefquad )
list(APPEND _cmake_import_check_files_for_sleef::sleefquad "${_IMPORT_PREFIX}/lib/libsleefquad.3.9.0.dylib" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
