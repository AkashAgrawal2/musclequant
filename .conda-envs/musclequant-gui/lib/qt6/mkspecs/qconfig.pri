host_build {
    QT_ARCH = x86_64
    QT_BUILDABI = 
    QT_TARGET_ARCH = arm64
    QT_TARGET_BUILDABI = 
} else {
    QT_ARCH = arm64
    QT_BUILDABI = 
    QT_LIBCPP_ABI_TAG = 
}
QT.global.enabled_features = version_tagging shared cross_compile rpath signaling_nan zstd thread future concurrent dbus openssl-linked opensslv30 test_gui shared cross_compile trivial_auto_var_init_pattern stack_protector libcpp_hardening shared rpath reduce_exports openssl
QT.global.disabled_features = static pkg-config debug_and_release separate_debug_info appstore-compliant simulator_and_device force_asserts framework c++20 c++2a c++2b c++2c reduce_relocations wasm-simd128 wasm-exceptions wasm-jspi opensslv11
QT.global.disabled_features += release build_all
QT_CONFIG += shared no-pkg-config rpath reduce_exports openssl release
CONFIG += release  shared cross_compile plugin_manifest trivial_auto_var_init_pattern stack_protector libcpp_hardening
QT_VERSION = 6.9.2
QT_MAJOR_VERSION = 6
QT_MINOR_VERSION = 9
QT_PATCH_VERSION = 2

QT_CLANG_MAJOR_VERSION = 19
QT_CLANG_MINOR_VERSION = 1
QT_CLANG_PATCH_VERSION = 7
QT_MAC_SDK_VERSION = 15.5
QMAKE_MACOSX_DEPLOYMENT_TARGET = 11.0
QT_MAC_SDK_VERSION_MIN = 14
QT_MAC_SDK_VERSION_MAX = 15
QT_ARCHS = $$QT_ARCH
