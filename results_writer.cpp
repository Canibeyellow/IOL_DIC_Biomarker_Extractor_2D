// =============================================================================
// results_writer.cpp
// =============================================================================

#include "results_writer.h"
#include "config.h"
#include "calibration_runtime.h"
#include <fstream>
#include <iostream>
#include <filesystem>
#include <iomanip>
#include <sstream>
#include <cmath>
#include <numeric>
#include <algorithm>
#include <ctime>
#include "compression_model.h"


namespace fs = std::filesystem;


// =============================================================================
// Internal helpers
// =============================================================================

static double pxToMm(double px)
{
    if (!g_calibration.valid) return 0.0;
    return px * g_calibration.pixel_size_mm_per_px;
}

// Returns "YYYYMMDD_HHMMSS" for the current local time.
static std::string processingTimestamp()
{
    std::time_t now = std::time(nullptr);
    std::tm     tm = {};
#ifdef _WIN32
    localtime_s(&tm, &now);
#else
    localtime_r(&now, &tm);
#endif
    std::ostringstream ss;
    ss << std::put_time(&tm, "%Y%m%d_%H%M%S");
    return ss.str();
}


// =============================================================================
// buildOutputFolder
// =============================================================================

std::string buildOutputFolder(const std::string& session_id)
{
    // e.g. D:\DIC_Results\20260510_174447_proc_20260607_143022
    std::string folder = Config::OUTPUT_FOLDER + "\\"
        + session_id
        + "_proc_"
        + processingTimestamp();
    fs::create_directories(folder);
    std::cout << "[RESULTS] Output folder : " << folder << "\n";
    return folder;
}


// =============================================================================
// writeFrameResults
// =============================================================================
// Columns (one row per frame):
//
//   IDENTITY
//     frame_index | timestamp | rolling_ref_idx | zncc_vs_rolling
//
//   TIMING
//     time_fftcc_sec | time_icgn_sec
//
//   QUALITY
//     valid_poi_count | mean_zncc
//
//   DISPLACEMENT — pixels
//     u_mean_px | v_mean_px | displacement_px
//
//   DISPLACEMENT — mm (blank if calibration unavailable)
//     u_mean_mm | v_mean_mm | displacement_mm
//
//   DECENTRATION
//     decentration_px | decentration_mm
//
//   ANGULAR
//     tilt_proxy_deg | rotation_deg
//
//   STATUS
//     bm_valid
//
//   CHAINED (only when Config::COMPUTE_CHAINED)
//     chain_u_mean_px | chain_v_mean_px | chain_displacement_px |
//     chain_decentration_px | chain_tilt_proxy_deg | chain_rotation_deg |
//     chain_valid

void writeFrameResults(const std::vector<FrameResult>& results,
    const std::string& output_folder)
{
    fs::create_directories(output_folder);
    std::string path = output_folder + "/frame_results.csv";
    std::ofstream f(path);

    if (!f.is_open())
        throw std::runtime_error("Cannot write: " + path);

    // ── Header ────────────────────────────────────────────────────────────────
    f << "frame_index,"
        << "timestamp,"
        << "rolling_ref_idx,"
        << "zncc_vs_rolling,"
        << "time_fftcc_sec,"
        << "time_icgn_sec,"
        << "valid_poi_count,"
        << "mean_zncc,"
        << "u_mean_px,"
        << "v_mean_px,"
        << "displacement_px,"
        << "u_mean_mm,"
        << "v_mean_mm,"
        << "displacement_mm,"
        << "decentration_px,"
        << "decentration_mm,"
        << "compression_mm,"          // always written, 0.0 in TRANSLATION mode
        << "u_mean_corrected_px,"
        << "v_mean_corrected_px,"
        << "decentration_corrected_px,"
        << "decentration_corrected_mm,"
        << "tilt_proxy_deg,"
        << "rotation_deg,"
        << "bm_valid";

    if (Config::COMPUTE_CHAINED) {
        f << ",chain_u_mean_px"
            << ",chain_v_mean_px"
            << ",chain_displacement_px"
            << ",chain_decentration_px"
            << ",chain_tilt_proxy_deg"
            << ",chain_rotation_deg"
            << ",chain_valid";
    }
    f << "\n";

    // ── Rows ──────────────────────────────────────────────────────────────────
    f << std::fixed << std::setprecision(6);

    for (const auto& r : results) {
        const auto& bm = r.bm_absolute;

        f << r.frame_index << ","
            << r.timestamp << ","
            << r.rolling_ref_idx << ","
            << r.zncc_vs_rolling << ","
            << r.time_fftcc_sec << ","
            << r.time_icgn_sec << ","
            << bm.valid_poi_count << ","
            << bm.mean_zncc << ","
            // Displacement — pixels
            << bm.u_mean << ","
            << bm.v_mean << ","
            << bm.displacement << ","
            // Displacement — mm
            << pxToMm(bm.u_mean) << ","
            << pxToMm(bm.v_mean) << ","
            << pxToMm(bm.displacement) << ","
            // Decentration
            << bm.decentration << ","
            << pxToMm(bm.decentration) << ","
            << r.compression_mm << ","
            << bm.u_mean_corrected << ","
            << bm.v_mean_corrected << ","
            << bm.decentration_corrected << ","
            << pxToMm(bm.decentration_corrected) << ","
            // Angular
            << bm.tilt_proxy_deg << ","
            << bm.rotation_deg << ","
            << (bm.valid ? "1" : "0");

        if (Config::COMPUTE_CHAINED && r.has_chained) {
            const auto& cb = r.bm_chained;
            f << "," << cb.u_mean
                << "," << cb.v_mean
                << "," << cb.displacement
                << "," << cb.decentration
                << "," << cb.tilt_proxy_deg
                << "," << cb.rotation_deg
                << "," << (cb.valid ? "1" : "0");
        }
        else if (Config::COMPUTE_CHAINED) {
            f << ",,,,,,,";
        }

        f << "\n";
    }

    f.close();
    std::cout << "[RESULTS] Frame results  → " << path << "\n";
}


// =============================================================================
// writeSummary
// =============================================================================
// Layout:
//   Section 1 — session metadata header
//   Section 2 — per-frame detail table (all biomarkers, raw values)
//   Section 3 — aggregate statistics (min / max / mean / std per biomarker)
//   Section 4 — final-state values (ISO-relevant)

void writeSummary(const std::vector<FrameResult>& results,
    const std::string& output_folder)
{
    fs::create_directories(output_folder);
    std::string path = output_folder + "/summary.csv";
    std::ofstream f(path);

    if (!f.is_open())
        throw std::runtime_error("Cannot write: " + path);

    f << std::fixed << std::setprecision(6);

    // ── Section 1: session metadata ───────────────────────────────────────────
    f << "session_id," << Config::SESSION_FOLDER << "\n";
    f << "output_folder," << output_folder << "\n";
    f << "frames_total," << results.size() << "\n";
    f << "session_mode,"
        << (Config::SESSION_MODE == Config::SessionMode::COMPRESSION
            ? "COMPRESSION" : "TRANSLATION") << "\n";
    f << "compression_step_mm," << Config::COMPRESSION_STEP_MM << "\n";
    f << "compression_from_metadata,"
        << (Config::COMPRESSION_FROM_METADATA ? "1" : "0") << "\n";
    f << "compression_symmetry_factor,"
        << Config::COMPRESSION_SYMMETRY_FACTOR << "\n";
    f << "calibration_src," << (g_calibration.valid
        ? g_calibration.source
        : "not loaded") << "\n";
    if (g_calibration.valid)
        f << "pixel_size_mm_per_px," << g_calibration.pixel_size_mm_per_px << "\n";
    f << "\n";

    // ── Section 2: per-frame detail ───────────────────────────────────────────
    f << "per_frame_results\n";
    f << "frame_index,"
        << "timestamp,"
        << "valid_poi_count,"
        << "mean_zncc,"
        << "u_mean_px,"
        << "v_mean_px,"
        << "displacement_px,"
        << "u_mean_mm,"
        << "v_mean_mm,"
        << "displacement_mm,"
        << "decentration_px,"
        << "decentration_mm,"
        << "tilt_proxy_deg,"
        << "rotation_deg,"
        << "bm_valid\n";

    for (const auto& r : results) {
        const auto& bm = r.bm_absolute;
        f << r.frame_index << ","
            << r.timestamp << ","
            << bm.valid_poi_count << ","
            << bm.mean_zncc << ","
            << bm.u_mean << ","
            << bm.v_mean << ","
            << bm.displacement << ","
            << pxToMm(bm.u_mean) << ","
            << pxToMm(bm.v_mean) << ","
            << pxToMm(bm.displacement) << ","
            << bm.decentration << ","
            << pxToMm(bm.decentration) << ","
            << bm.tilt_proxy_deg << ","
            << bm.rotation_deg << ","
            << (bm.valid ? "1" : "0") << "\n";
    }
    f << "\n";

    // ── Section 3: aggregate statistics ──────────────────────────────────────
    // Collect valid-frame vectors
    std::vector<double> u_px, v_px, disp_px, decent_px, tilt, rot;
    std::vector<double> u_mm, v_mm, disp_mm, decent_mm;

    for (const auto& r : results) {
        if (!r.bm_absolute.valid) continue;
        const auto& bm = r.bm_absolute;
        u_px.push_back(bm.u_mean);
        v_px.push_back(bm.v_mean);
        disp_px.push_back(bm.displacement);
        decent_px.push_back(bm.decentration);
        tilt.push_back(bm.tilt_proxy_deg);
        rot.push_back(bm.rotation_deg);
        u_mm.push_back(pxToMm(bm.u_mean));
        v_mm.push_back(pxToMm(bm.v_mean));
        disp_mm.push_back(pxToMm(bm.displacement));
        decent_mm.push_back(pxToMm(bm.decentration));
    }

    auto stats = [](const std::vector<double>& v,
        double& mn, double& mx, double& mean, double& sd)
        {
            if (v.empty()) { mn = mx = mean = sd = 0.0; return; }
            mn = *std::min_element(v.begin(), v.end());
            mx = *std::max_element(v.begin(), v.end());
            mean = std::accumulate(v.begin(), v.end(), 0.0) / v.size();
            double var = 0.0;
            for (auto x : v) var += (x - mean) * (x - mean);
            sd = std::sqrt(var / v.size());
        };

    double mn, mx, mean, sd;

    auto writeRow = [&](const std::string& name,
        const std::vector<double>& vals,
        const std::string& unit)
        {
            stats(vals, mn, mx, mean, sd);
            f << name << ","
                << mn << ","
                << mx << ","
                << mean << ","
                << sd << ","
                << unit << ","
                << vals.size() << "\n";
        };

    f << "aggregate_statistics\n";
    f << "biomarker,min,max,mean,std,unit,valid_frames\n";

    // Pixels
    writeRow("u_mean_px", u_px, "pixels");
    writeRow("v_mean_px", v_px, "pixels");
    writeRow("displacement_px", disp_px, "pixels");
    writeRow("decentration_px", decent_px, "pixels");
    writeRow("tilt_proxy_deg", tilt, "degrees");
    writeRow("rotation_deg", rot, "degrees");

    if (Config::SESSION_MODE == Config::SessionMode::COMPRESSION) {
        // Collect corrected decentration
        std::vector<double> decent_corr_px, decent_corr_mm;
        for (const auto& r : results) {
            if (!r.bm_absolute.valid) continue;
            decent_corr_px.push_back(r.bm_absolute.decentration_corrected);
            decent_corr_mm.push_back(pxToMm(r.bm_absolute.decentration_corrected));
        }
        writeRow("decentration_corrected_px", decent_corr_px, "pixels");
        if (g_calibration.valid)
            writeRow("decentration_corrected_mm", decent_corr_mm, "mm");
    }

    // mm (written but zeroed if calibration unavailable)
    if (g_calibration.valid) {
        writeRow("u_mean_mm", u_mm, "mm");
        writeRow("v_mean_mm", v_mm, "mm");
        writeRow("displacement_mm", disp_mm, "mm");
        writeRow("decentration_mm", decent_mm, "mm");
    }
    else {
        f << "# mm columns omitted — calibration not loaded\n";
    }

    f << "\n";

    // ── Section 4: final-state values (ISO-relevant) ──────────────────────────
    f << "final_state_values\n";
    if (!results.empty() && results.back().bm_absolute.valid) {
        const auto& last = results.back().bm_absolute;
        f << "u_mean_final_px," << last.u_mean << "\n";
        f << "v_mean_final_px," << last.v_mean << "\n";
        f << "displacement_final_px," << last.displacement << "\n";
        f << "decentration_final_px," << last.decentration << "\n";
        if (g_calibration.valid) {
            f << "u_mean_final_mm," << pxToMm(last.u_mean) << "\n";
            f << "v_mean_final_mm," << pxToMm(last.v_mean) << "\n";
            f << "displacement_final_mm," << pxToMm(last.displacement) << "\n";
            f << "decentration_final_mm," << pxToMm(last.decentration) << "\n";
        }
        f << "tilt_final_deg," << last.tilt_proxy_deg << "\n";
        f << "rotation_final_deg," << last.rotation_deg << "\n";
    }
    else {
        f << "final_state_valid,0\n";
    }

    f.close();
    std::cout << "[RESULTS] Summary        → " << path << "\n";
}


void writeBehaviourSummary(const std::vector<FrameResult>& results,
    const std::string& output_folder)
{
    fs::create_directories(output_folder);
    std::string path = output_folder + "/behaviour_summary.txt";
    std::ofstream f(path);
    if (!f.is_open())
        throw std::runtime_error("Cannot write: " + path);

    std::vector<double> comp, dec, decc, tilt, rot;
    int n_bm_valid = 0;
    for (const auto& r : results) {
        const auto& bm = r.bm_absolute;
        if (!bm.valid) continue;
        ++n_bm_valid;
        comp.push_back(r.compression_mm);
        dec.push_back(pxToMm(bm.decentration));
        decc.push_back(pxToMm(bm.decentration_corrected));
        tilt.push_back(bm.tilt_proxy_deg);
        rot.push_back(bm.rotation_deg);
    }

    auto meanv = [&](const std::vector<double>& v) {return v.empty() ? 0.0 : std::accumulate(v.begin(), v.end(), 0.0) / v.size(); };
    auto minv = [&](const std::vector<double>& v) {return v.empty() ? 0.0 : *std::min_element(v.begin(), v.end()); };
    auto maxv = [&](const std::vector<double>& v) {return v.empty() ? 0.0 : *std::max_element(v.begin(), v.end()); };
    auto stdv = [&](const std::vector<double>& v) { if (v.size() < 2) return 0.0; double m = meanv(v), s = 0; for (double x : v)s += (x - m) * (x - m); return std::sqrt(s / v.size()); };
    auto slope = [&](const std::vector<double>& y, bool& ok) { ok = false; if (comp.size() < 2) return 0.0;
    double mx = meanv(comp), my = meanv(y), sxx = 0, sxy = 0;
    for (size_t i = 0; i < comp.size(); ++i) { sxx += (comp[i] - mx) * (comp[i] - mx); sxy += (comp[i] - mx) * (y[i] - my); }
    if (sxx < 1e-9) return 0.0; ok = true; return sxy / sxx; };

    int idx_max = -1; double best = -1e30;
    for (size_t i = 0; i < results.size(); ++i) {
        if (!results[i].bm_absolute.valid) continue;
        if (results[i].compression_mm >= best) { best = results[i].compression_mm; idx_max = (int)i; }
    }

    std::string date = results.empty() ? "(none)" : results.front().timestamp;

    f << "===============================================================\n";
    f << "  IOL BEHAVIOUR SUMMARY  (full-field pipeline)\n";
    f << "===============================================================\n";
    f << "Date (first frame) : " << date << "\n";
    f << "Frames total       : " << results.size() << "\n";
    f << "Frames bm-valid    : " << n_bm_valid << "\n";
    f << std::fixed << std::setprecision(3);
    f << "Compression range  : " << minv(comp) << " -> " << maxv(comp) << " mm\n";
    f << std::setprecision(6);
    f << "Pixel size         : " << (g_calibration.valid ? g_calibration.pixel_size_mm_per_px : 0.0)
        << " mm/px" << (g_calibration.valid ? "" : "  (UNCALIBRATED)") << "\n\n";

    bool okd, okdc, okt, okr;
    double sld = slope(dec, okd), sldc = slope(decc, okdc), slt = slope(tilt, okt), slr = slope(rot, okr);
    auto row = [&](const std::string& name, const std::vector<double>& v, double atmax, double sl, bool ok, const std::string& note) {
        f << std::left << std::setw(24) << name << std::right << std::fixed << std::setprecision(4)
            << std::setw(11) << minv(v) << std::setw(11) << maxv(v) << std::setw(11) << meanv(v)
            << std::setw(11) << stdv(v) << std::setw(13) << atmax;
        if (ok) f << std::setw(13) << sl; else f << std::setw(13) << "N/A";
        f << "   " << note << "\n"; };

    f << "BIOMARKER BEHAVIOUR  (bm-valid frames only)\n";
    f << "---------------------------------------------------------------\n";
    f << std::left << std::setw(24) << "biomarker" << std::right
        << std::setw(11) << "min" << std::setw(11) << "max" << std::setw(11) << "mean"
        << std::setw(11) << "std" << std::setw(13) << "at_max_comp" << std::setw(13) << "slope/mm" << "\n";
    row("decentration_mm", dec, idx_max >= 0 ? pxToMm(results[idx_max].bm_absolute.decentration) : 0.0, sld, okd, "");
    row("decentration_corr_mm", decc, idx_max >= 0 ? pxToMm(results[idx_max].bm_absolute.decentration_corrected) : 0.0, sldc, okdc, "compression-corrected");
    row("tilt_proxy_deg", tilt, idx_max >= 0 ? results[idx_max].bm_absolute.tilt_proxy_deg : 0.0, slt, okt, "2D in-plane proxy");
    row("rotation_deg", rot, idx_max >= 0 ? results[idx_max].bm_absolute.rotation_deg : 0.0, slr, okr, "+ = CCW");
    f << "\n";

    f << "FINAL STATE (frame at max compression)\n";
    f << "---------------------------------------------------------------\n";
    if (idx_max >= 0) {
        const auto& bm = results[idx_max].bm_absolute;
        f << std::setprecision(3);
        f << "  compression  : " << results[idx_max].compression_mm << " mm\n";
        f << std::setprecision(4);
        f << "  decentration : " << pxToMm(bm.decentration) << " mm  (corr "
            << pxToMm(bm.decentration_corrected) << " mm)\n";
        f << "  tilt (proxy) : " << bm.tilt_proxy_deg << " deg\n";
        f << "  rotation     : " << bm.rotation_deg << " deg\n";
    }
    else f << "  (no biomarker-valid frame)\n";
    f << "\nNotes: decentration = sqrt(u^2+v^2) of the field; corrected = minus the\n";
    f << "compression rigid-body component; tilt is a 2D in-plane proxy; rotation\n";
    f << "+ = counter-clockwise; slope/mm = LSQ rate vs compression (N/A when flat).\n";

    f.close();
    std::cout << "[RESULTS] behaviour_summary.txt -> " << path << "\n";
}

void writeDatasetQuality(const std::vector<FrameResult>& results,
    const std::string& output_folder)
{
    fs::create_directories(output_folder);
    std::string path = output_folder + "/dataset_quality.txt";
    std::ofstream f(path);
    if (!f.is_open())
        throw std::runtime_error("Cannot write: " + path);

    const int nf = (int)results.size();
    int n_bm_valid = 0;
    std::vector<double> poi, zncc, roll, tt;
    for (const auto& r : results) {
        const auto& bm = r.bm_absolute;
        if (bm.valid) ++n_bm_valid;
        poi.push_back((double)bm.valid_poi_count);
        zncc.push_back(bm.mean_zncc);
        roll.push_back(r.zncc_vs_rolling);
        tt.push_back(r.time_fftcc_sec + r.time_icgn_sec);
    }
    auto meanv = [&](const std::vector<double>& v) {return v.empty() ? 0.0 : std::accumulate(v.begin(), v.end(), 0.0) / v.size(); };
    auto minv = [&](const std::vector<double>& v) {return v.empty() ? 0.0 : *std::min_element(v.begin(), v.end()); };

    double bm_valid_frac = nf ? (double)n_bm_valid / nf : 0.0;
    double overall_zncc = meanv(zncc);
    const char* tier = "POOR";
    if (bm_valid_frac >= Config::QUALITY_GOOD_FRAME_VALID_FRAC && overall_zncc >= Config::QUALITY_GOOD_MEAN_ZNCC) tier = "GOOD";
    else if (bm_valid_frac >= Config::QUALITY_OKAY_FRAME_VALID_FRAC && overall_zncc >= Config::QUALITY_OKAY_MEAN_ZNCC) tier = "OKAY";

    f << std::fixed;
    f << "===============================================================\n";
    f << "  DATASET QUALITY & RELIABILITY  (full-field pipeline)\n";
    f << "===============================================================\n";
    f << "Frames total      : " << nf << "\n";
    f << "Biomarker-valid   : " << n_bm_valid << "  ("
        << std::setprecision(1) << 100.0 * bm_valid_frac << "%)\n\n";
    f << std::setprecision(3);
    f << "VALID POI COUNT     min/mean : " << minv(poi) << " / " << meanv(poi) << "\n";
    f << "MEAN ZNCC (vs ref)  min/mean : " << minv(zncc) << " / " << meanv(zncc) << "\n";
    f << "ZNCC vs ROLLING REF min/mean : " << minv(roll) << " / " << meanv(roll) << "\n\n";
    f << "TIMING\n";
    f << "---------------------------------------------------------------\n";
    f << "  total      : " << std::accumulate(tt.begin(), tt.end(), 0.0) << " s\n";
    f << "  mean/frame : " << meanv(tt) << " s\n\n";
    f << "RATING INPUTS  (you decide; thresholds in config.h)\n";
    f << "---------------------------------------------------------------\n";
    f << "  bm-valid fraction : " << bm_valid_frac << "\n";
    f << "  overall mean ZNCC : " << overall_zncc << "\n";
    f << "  suggested tier    : " << tier << "   (heuristic, not a verdict)\n";
    f << "\nNote: full-field reliability is grid-based (valid POI count + mean ZNCC),\n";
    f << "not per-ROI; zncc_vs_rolling tracks rolling-reference health.\n";

    f.close();
    std::cout << "[RESULTS] dataset_quality.txt -> " << path << "\n";
}

// =============================================================================
// cleanupPreparedImages
// =============================================================================

void cleanupPreparedImages(const std::string& output_folder)
{
    fs::path prepared_dir = fs::path(output_folder) / "prepared";

    if (!fs::exists(prepared_dir)) return;

    std::error_code ec;
    std::uintmax_t  removed = fs::remove_all(prepared_dir, ec);

    if (ec)
        std::cout << "[CLEANUP] WARNING: could not remove prepared/ : "
        << ec.message() << "\n";
    else
        std::cout << "[CLEANUP] Removed prepared/ ("
        << removed << " files).\n";
}