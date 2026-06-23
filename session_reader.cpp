// =============================================================================
// session_reader.cpp
// =============================================================================

#include "session_reader.h"
#include "config.h"
#include <fstream>
#include <iostream>
#include <filesystem>
#include <stdexcept>
#include <algorithm>
#include <nlohmann/json.hpp>
#include <cctype>
namespace fs = std::filesystem;
using json   = nlohmann::json;


// =============================================================================
// Internal helpers
// =============================================================================

// Resolves a path from metadata.json.
// The Pi writes absolute Pi paths (e.g. /home/pi/dic_acquisition/sessions/...).
// After WinSCP copies the folder, those paths no longer exist on Windows.
// We fix this by replacing the Pi prefix with the local SESSION_FOLDER root.
static std::string resolvePath(const std::string& pi_path,
                               const std::string& session_root)
{
    // Find the session ID portion of the pi_path (e.g. "20260330_142500")
    // and replace everything before it with the local session_root.
    // The session ID is the last component of session_root.
    std::string session_id = fs::path(session_root).filename().string();

    auto pos = pi_path.find(session_id);
    if (pos == std::string::npos) {
        // Cannot resolve — return as-is and let the caller handle the error
        return pi_path;
    }

    // Everything from session_id onward is the relative path
    std::string relative = pi_path.substr(pos + session_id.size());

    // Replace forward slashes for Windows compatibility
    std::replace(relative.begin(), relative.end(), '/', '\\');

    return session_root + relative;
}


// Loads all *.bmp (or *.tif / *.png) paths from a folder, sorted by name.
// Returns lowercase file extension: ".bmp", ".png", ".tif", etc.
static std::string lowerExt(const fs::path& p)
{
    std::string e = p.extension().string();
    std::transform(e.begin(), e.end(), e.begin(), ::tolower);
    return e;
}


// Loads all supported image paths from a folder, sorted by name.
static std::vector<std::string> globFolder(const std::string& folder)
{
    std::vector<std::string> paths;

    if (!fs::is_directory(folder))
        return paths;

    for (const auto& entry : fs::directory_iterator(folder)) {
        if (!entry.is_regular_file())
            continue;

        std::string e = lowerExt(entry.path());

        if (e == ".bmp" || e == ".png" || e == ".tif" || e == ".tiff") {
            paths.push_back(entry.path().string());
        }
    }

    std::sort(paths.begin(), paths.end());
    return paths;
}


// =============================================================================
// loadSession
// =============================================================================

SessionData loadSession(const std::string& session_folder)
{
    SessionData sd;
    sd.session_id = fs::path(session_folder).filename().string();

    // ── Priority 1: manual override ──────────────────────────────────────────
    if (!Config::MANUAL_REF_PATH.empty()) {
        std::cout << "[SESSION] Using manual reference path.\n";
        sd.ref_path = Config::MANUAL_REF_PATH;

        if (!Config::MANUAL_FRAME_GLOB.empty()) {
            // Treat MANUAL_FRAME_GLOB as a folder path and glob it
            auto paths = globFolder(Config::MANUAL_FRAME_GLOB);
            int idx = 0;
            for (const auto& p : paths) {
                FrameRecord fr;
                fr.frame_index     = ++idx;
                fr.path            = p;
                fr.rolling_ref_idx = 0;   // no metadata — assume frame 0
                fr.timestamp = "";
                fr.zncc_vs_rolling = 0.0;
                fr.rolling_updated = false;
                sd.frames.push_back(fr);
            }
        }

        if (!fs::exists(sd.ref_path))
            throw std::runtime_error("Manual ref path not found: " + sd.ref_path);

        printSessionSummary(sd);
        return sd;
    }

    // ── Priority 2: parse metadata.json ──────────────────────────────────────
    std::string meta_path = session_folder + "/metadata.json";
    if (!fs::exists(meta_path)) {
        throw std::runtime_error(
            "metadata.json not found at: " + meta_path +
            "\nEither set MANUAL_REF_PATH in config.h or copy the full "
            "session folder from the Pi."
        );
    }

    else {
        std::cout << "[SESSION] No compression log in metadata — "
            << "using constant step assumption ("
            << Config::COMPRESSION_STEP_MM << "mm/frame)\n";
    }

    std::ifstream f(meta_path);
    if (!f.is_open())
        throw std::runtime_error("Cannot open: " + meta_path);

    json meta = json::parse(f);

    // ── Extract session config ────────────────────────────────────────────────
    if (meta.contains("config")) {
        auto& cfg = meta["config"];
        sd.zncc_threshold = cfg.value("zncc_threshold", 0.6);
        sd.capture_width  = cfg.value("capture_width",  0);
        sd.capture_height = cfg.value("capture_height", 0);
    }

    // Warn if Pi-side speckle was also applied — we will apply again on desktop.

    // ── Extract reference path ────────────────────────────────────────────────
    if (!meta.contains("reference") || meta["reference"].is_null()) {
        throw std::runtime_error(
            "metadata.json has no reference entry. "
            "Did you press R to set the reference during capture?"
        );
    }

    std::string pi_ref = meta["reference"]["prepared"];
    sd.ref_path = resolvePath(pi_ref, session_folder);

    if (!fs::exists(sd.ref_path)) {
        throw std::runtime_error(
            "Reference image not found after path resolution:\n"
            "  Pi path   : " + pi_ref + "\n"
            "  Resolved  : " + sd.ref_path + "\n"
            "Check SESSION_FOLDER in config.h matches the folder WinSCP copied."
        );
    }

    // ── Extract deformed frames ───────────────────────────────────────────────
    if (!meta.contains("frames") || meta["frames"].empty()) {
        throw std::runtime_error(
            "metadata.json has no frames. "
            "No deformed frames were captured in this session."
        );
    }

    for (const auto& entry : meta["frames"]) {
        FrameRecord fr;
        fr.frame_index     = entry.value("frame_index",         0);
        fr.timestamp       = entry.value("timestamp",           "");
        fr.zncc_vs_rolling = entry.value("zncc_vs_rolling_ref", 0.0);
        fr.rolling_ref_idx = entry.value("rolling_ref_idx",     0);
        fr.rolling_updated = entry.value("rolling_ref_updated", false);

        std::string pi_path = entry.value("path", "");
        fr.path = resolvePath(pi_path, session_folder);

        if (!fs::exists(fr.path)) {
            std::cerr << "[SESSION] WARNING: frame file not found, skipping: "
                      << fr.path << "\n";
            continue;
        }

        sd.frames.push_back(fr);
    }

    if (sd.frames.empty()) {
        throw std::runtime_error(
            "No valid frame files found after resolving metadata paths. "
            "Check SESSION_FOLDER in config.h."
        );
    }
    // Optional compression log from Pico motor controller
    if (meta.contains("compression_log") && meta["compression_log"].is_array()) {
        for (const auto& val : meta["compression_log"]) {
            sd.compression_log.push_back(val.get<double>());
        }
        std::cout << "[SESSION] Compression log: "
            << sd.compression_log.size() << " entries\n";
    }
    else {
        std::cout << "[SESSION] No compression log in metadata — "
            << "using constant step assumption ("
            << Config::COMPRESSION_STEP_MM << "mm/frame)\n";
    }

    printSessionSummary(sd);
    return sd;
}


// =============================================================================
// printSessionSummary
// =============================================================================

void printSessionSummary(const SessionData& session)
{
    std::cout << "\n[SESSION] ── Session loaded ─────────────────────────────\n";
    std::cout << "[SESSION] ID          : " << session.session_id        << "\n";
    std::cout << "[SESSION] Reference   : " << session.ref_path          << "\n";
    std::cout << "[SESSION] Frames      : " << session.frames.size()     << "\n";
    std::cout << "[SESSION] Image size  : " << session.capture_width
              << " x "                      << session.capture_height    << "\n";

    // Print rolling reference update events
    int updates = 0;
    for (const auto& fr : session.frames)
        if (fr.rolling_updated) ++updates;

    std::cout << "[SESSION] Roll.ref upd: " << updates << " times\n";
    std::cout << "[SESSION] ──────────────────────────────────────────────\n\n";
}
