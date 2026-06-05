add_executable(optics_tests test_main.cpp)
target_link_libraries(optics_tests PRIVATE optics::optics)

target_compile_options(optics_tests PRIVATE
    $<$<CXX_COMPILER_ID:MSVC>:/W4>
    $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-Wall -Wextra>
    # The suite asserts via assert(); undefine NDEBUG (injected in optimized
    # configs) on this target so assertions stay live. Applied after config
    # flags, so it wins.
    $<$<CXX_COMPILER_ID:MSVC>:/UNDEBUG>
    $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-UNDEBUG>)

add_test(NAME optics_tests COMMAND optics_tests)
