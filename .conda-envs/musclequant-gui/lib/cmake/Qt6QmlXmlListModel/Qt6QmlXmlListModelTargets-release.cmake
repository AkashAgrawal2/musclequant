#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "Qt6::QmlXmlListModel" for configuration "Release"
set_property(TARGET Qt6::QmlXmlListModel APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(Qt6::QmlXmlListModel PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "Qt6::QmlModels"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libQt6QmlXmlListModel.6.9.2.dylib"
  IMPORTED_SONAME_RELEASE "@rpath/libQt6QmlXmlListModel.6.dylib"
  )

list(APPEND _cmake_import_check_targets Qt6::QmlXmlListModel )
list(APPEND _cmake_import_check_files_for_Qt6::QmlXmlListModel "${_IMPORT_PREFIX}/lib/libQt6QmlXmlListModel.6.9.2.dylib" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
