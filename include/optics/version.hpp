// Copyright Ingo Proff 2016.
// https://github.com/J-D-3/density-clustering
// Distributed under the MIT Software License (X11 license).
// (See accompanying file LICENSE)

#pragma once

// Library version. Keep in sync with the project() VERSION in CMakeLists.txt.
#define OPTICS_VERSION_MAJOR 1
#define OPTICS_VERSION_MINOR 0
#define OPTICS_VERSION_PATCH 0
#define OPTICS_VERSION_STRING "1.0.0"

// Single integer for easy comparisons: major*10000 + minor*100 + patch.
#define OPTICS_VERSION ( OPTICS_VERSION_MAJOR * 10000 + OPTICS_VERSION_MINOR * 100 + OPTICS_VERSION_PATCH )

namespace optics {
inline constexpr const char* version() { return OPTICS_VERSION_STRING; }
}  // namespace optics
