# Example programs. Built when OPTICS_BUILD_EXAMPLES is ON (default when this is
# the top-level project). They link the header-only library like any consumer.

add_executable(optics_color color_clustering/optics_color.cpp)
target_link_libraries(optics_color PRIVATE optics::optics)
