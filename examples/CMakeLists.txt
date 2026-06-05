# Example programs. Built when OPTICS_BUILD_EXAMPLES is ON (default when this is
# the top-level project). They link the header-only library like any consumer.

add_executable(optics_color color_clustering/optics_color.cpp)
target_link_libraries(optics_color PRIVATE optics::optics)

# Generic "cluster your own CSV" example (any of dim 2/3/4/16). Pairs with
# tools/datasets.py (synthetic 2-D sets) and tools/visualize.py (plots).
add_executable(cluster_csv cluster_csv/cluster_csv.cpp)
target_link_libraries(cluster_csv PRIVATE optics::optics)
