#pragma once

#include <filesystem>
#include <string>

namespace ADU {

struct AppContext;

/// Command-line parsed paths for config and resources.
struct AppInitParams {
    std::string config_path;
    std::filesystem::path resource_path;
};

/// Parse argv into config path and resource path. Throws on error.
AppInitParams parse_args(int argc, const char* argv[]);

/// Load config, meshes, solver, constraints, BVH. Populates ctx.
void init_app(AppContext& ctx, const AppInitParams& params);

} // namespace ADU
