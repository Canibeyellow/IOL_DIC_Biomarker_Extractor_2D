#include "optic_runtime.h"
#include "config.h"

#include <iostream>
#include <fstream>
#include <filesystem>
#include <nlohmann/json.hpp>
#include "calibration_runtime.h"

namespace fs = std::filesystem;
using json = nlohmann::json;

RuntimeOptic g_optic;

void loadOptic(const std::string& session_folder)
{
    g_optic.valid = false;
    if (!Config::USE_OPTIC_DETECTION) {
        std::cout << "[OPTIC] Detection disabled in config.\n";
        return;
    }

    fs::path p = fs::path(session_folder) / "optic_latest.json";
    std::ifstream f(p);
    if (!f.is_open()) {
        std::cout << "[OPTIC] No optic_latest.json — using static ROI.\n";
        return;
    }

    json j = json::parse(f, nullptr, false);
    if (j.is_discarded() || j.value("used_fallback", true)) {
        std::cout << "[OPTIC] Detection marked fallback — using static ROI.\n";
        return;
    }

    g_optic.valid = true;
    g_optic.cx = j.value("cx", 0);
    g_optic.cy = j.value("cy", 0);
    g_optic.radius_px = j.value("radius_px", 0.0);
    g_optic.confidence = j.value("confidence", 0.0);
    g_optic.source = p.string();
    std::cout << "[OPTIC] Loaded: centre=(" << g_optic.cx << "," << g_optic.cy
        << ") R=" << g_optic.radius_px
        << " conf=" << g_optic.confidence << "\n";
}

EffectiveROI resolveEffectiveROI()
{
    EffectiveROI e;
    e.cx = Config::ROI_CENTER_X;
    e.cy = Config::ROI_CENTER_Y;

    if (Config::ROI_USE_PHYSICAL_SIZE && g_calibration.valid)
        e.radius_px = (int)(Config::ROI_RADIUS_MM / g_calibration.pixel_size_mm_per_px);
    else
        e.radius_px = Config::ROI_RADIUS_PX_FALLBACK;

    if (Config::USE_OPTIC_DETECTION && g_optic.valid) {
        e.cx = g_optic.cx;
        e.cy = g_optic.cy;
        e.radius_px = (int)g_optic.radius_px;
        e.from_detection = true;
    }
    return e;
}