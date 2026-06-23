// =============================================================================
// main.cpp
// DIC Processor — Entry Point
// =============================================================================
// Supports two processing architectures, selected by Config::ROI_BASED_MODE:
//
//   ROI_BASED_MODE = false  (default)
//     Full-field DIC — original architecture.
//     3020 POIs across the full circular optic ROI.
//     Outputs: frame_results.csv, summary.csv
//
//   ROI_BASED_MODE = true
//     Five-ROI DIC — ISO-aligned P/Q/R/S/C architecture.
//     Five independent DIC instances on small patches at ISO landmark positions.
//     Biomarkers computed from geometric point fitting, not field averaging.
//     Outputs: roi_frame_results.csv, roi_summary.csv
//
// Both modes share:
//   - Session loading (metadata.json)
//   - Image preparation (speckle overlay / copy to prepared/)
//   - Calibration loading
//   - Output folder management
//   - Cleanup of prepared/ on exit
//
// To switch modes: change ROI_BASED_MODE in config.h and rebuild.
// Both paths are always compiled — no conditional compilation.
//
// DEPENDENCIES:
//   OpenCorr      — opencorr.h
//   OpenCV 4.x    — image loading, speckle overlay
//   Eigen 3       — used inside OpenCorr
//   FFTW3         — used inside OpenCorr FFTCC
//   nlohmann/json — json.hpp (single header)
//
// BUILD (Visual Studio):
//   Add to project: all .cpp files in this folder
//   Include dirs:   OpenCorr/src, eigen3, opencv/include, $(ProjectDir)
//   Libs:           opencv_world4xx.lib, libfftw3f-3.lib
//   DLLs to copy:   opencv_world4xx.dll, libfftw3f-3.dll
// =============================================================================

#include <iostream>
#include <filesystem>
#include <stdexcept>
#include <iomanip>
#include <omp.h>
#include <vector>
#include <fstream>

#include "config.h"
#include "session_reader.h"
#include "calibration_runtime.h"
#include "compression_model.h"
#include "opencorr.h"
#include <opencv2/opencv.hpp>
#include <nlohmann/json.hpp>
#include  "optic_runtime.h"

// Full-field pipeline headers
#include "dic_pipeline.h"
#include "biomarkers.h"
#include "results_writer.h"

// ROI-based pipeline headers
#include "dic_pipeline_roi.h"
#include "biomarkers_roi.h"
#include "results_writer_roi.h"

namespace fs = std::filesystem;
using namespace opencorr;


// =============================================================================
// prepareImageFile
// =============================================================================
// Loads an image as grayscale and writes it to output_folder/prepared/.
// OpenCorr's Image2D requires a file path — it cannot accept a cv::Mat.
// Writing to a canonical prepared/ folder means both pipeline modes always
// read from the same location, simplifying path management.

static std::string prepareImageFile(const std::string& input_path,
    const std::string& output_folder)
{
    cv::Mat gray = cv::imread(input_path, cv::IMREAD_GRAYSCALE);
    if (gray.empty())
        throw std::runtime_error("Cannot load image: " + input_path);

    fs::create_directories(output_folder + "/prepared");

    std::string filename = fs::path(input_path).stem().string() + "_prepared.bmp";
    std::string out_path = output_folder + "/prepared/" + filename;

    if (!cv::imwrite(out_path, gray))
        throw std::runtime_error("Cannot write prepared image: " + out_path);

    return out_path;
}


// =============================================================================
// printStartBanner
// =============================================================================

static void printStartBanner(const std::string& session_folder,
    bool roi_mode)
{
    std::cout << "=====================================================\n"
        << "  DIC PROCESSOR — IOL Biomarker Extraction\n"
        << "  Session : " << session_folder << "\n"
        << "  Mode    : "
        << (roi_mode ? "ROI-BASED (5-point ISO)" : "FULL-FIELD")
        << "\n"
        << "=====================================================\n\n";
}


// =============================================================================
// printFinalSummary — full-field mode
// =============================================================================

static void printFinalSummaryFullField(const std::vector<FrameResult>& results,
    const std::string& output_folder)
{
    std::cout << "\n=====================================================\n";
    std::cout << "  PROCESSING COMPLETE  (FULL-FIELD)\n";
    std::cout << "  Frames processed : " << results.size() << "\n";

    // Find last valid frame
    const FrameResult* last = nullptr;
    for (int i = (int)results.size() - 1; i >= 0; --i) {
        if (results[i].bm_absolute.valid) { last = &results[i]; break; }
    }

    if (last) {
        const auto& bm = last->bm_absolute;
        std::cout << std::fixed << std::setprecision(4);
        std::cout << "  ── Final state (absolute reference) ─────────────\n";
        std::cout << "  Displacement  : " << bm.displacement << " px\n";
        std::cout << "  Decentration  : " << bm.decentration << " px\n";
        std::cout << "  Tilt proxy 2D : " << bm.tilt_proxy_deg << " deg\n";
        std::cout << "  Rotation      : " << bm.rotation_deg << " deg\n";

        if (g_calibration.valid) {
            double sc = g_calibration.pixel_size_mm_per_px;
            std::cout << "  ── Physical units (" << g_calibration.source << ")\n";
            std::cout << "  Scale         : " << sc << " mm/px\n";
            std::cout << "  Displacement  : " << bm.displacement * sc << " mm\n";
            std::cout << "  Decentration  : " << bm.decentration * sc << " mm\n";
        }
        else {
            std::cout << "  [Physical units unavailable — calibration not loaded]\n";
        }
    }
    std::cout << "  Results folder : " << output_folder << "\n";
    std::cout << "=====================================================\n\n";
}


// =============================================================================
// printFinalSummary — ROI mode
// =============================================================================

static void printFinalSummaryROI(const std::vector<ROIFrameResult>& results,
    const std::string& output_folder)
{
    std::cout << "\n=====================================================\n";
    std::cout << "  PROCESSING COMPLETE  (ROI-BASED)\n";
    std::cout << "  Frames processed : " << results.size() << "\n";

    // Find last valid frame
    const ROIFrameResult* last = nullptr;
    for (int i = (int)results.size() - 1; i >= 0; --i) {
        if (results[i].bm.valid) { last = &results[i]; break; }
    }

    if (last) {
        const auto& bm = last->bm;
        std::cout << std::fixed << std::setprecision(4);
        std::cout << "  ── Final state (ISO P/Q/R/S/C) ──────────────────\n";
        std::cout << "  Decentration  : " << bm.dec_px << " px";
        if (g_calibration.valid)
            std::cout << "  (" << bm.dec_mm << " mm)";
        std::cout << "\n";
        std::cout << "  Tilt          : " << bm.tilt_deg << " deg\n";
        std::cout << "  Rotation      : " << bm.rotation_deg << " deg\n";
        if (last->dic.interpolated_roi_count > 0)
            std::cout << "  NOTE: " << last->dic.interpolated_roi_count
            << " ROI(s) interpolated in final frame\n";
    }
    std::cout << "  Results folder : " << output_folder << "\n";
    std::cout << "=====================================================\n\n";
}


// =============================================================================
// runFullFieldPipeline
// =============================================================================
// Original full-field DIC path — unchanged logic, extracted into a function
// for clarity.

static void runFullFieldPipeline(
    const SessionData& session,
    const std::string& ref_prepared,
    const std::vector<std::string>& prepared_paths,
    const std::string& output_folder)
{
    // ── RAII guard — destroyDICContext even on exception ──────────────────────
    struct DICContextGuard {
        DICContext ctx;
        bool       armed = false;
        ~DICContextGuard() { if (armed) destroyDICContext(ctx); }
    };

    std::cout << "[MAIN]   Building full-field DIC context ...\n";
    DICContextGuard guard;
    guard.ctx = createDICContext(ref_prepared);
    guard.armed = true;
    DICContext& dic_ctx = guard.ctx;
    std::cout << "[MAIN]   DIC context ready.\n\n";

    std::vector<FrameResult> all_results;
    all_results.reserve(session.frames.size());

    for (size_t i = 0; i < session.frames.size(); ++i) {
        double frame_t0 = omp_get_wtime();
        const FrameRecord& fr = session.frames[i];

        FrameResult result;
        result.frame_index = fr.frame_index;
        result.timestamp = fr.timestamp;
        result.rolling_ref_idx = fr.rolling_ref_idx;
        result.zncc_vs_rolling = fr.zncc_vs_rolling;

        // ── Absolute reference DIC ────────────────────────────────────────────
        DICResult dic_abs = runDIC(dic_ctx, prepared_paths[i], fr.frame_index);

        result.time_fftcc_sec = dic_abs.time_fftcc_sec;
        result.time_icgn_sec = dic_abs.time_icgn_sec;

        if (Config::SAVE_DIC_MAPS || Config::SAVE_POI_TABLES)
            saveDICResult(dic_abs, prepared_paths[i],
                Config::OUTPUT_FOLDER, fr.frame_index);

        // ── Biomarkers ────────────────────────────────────────────────────────
        double bm_t0 = omp_get_wtime();
        double comp_mm = getCompressionMm(session, fr.frame_index);

        result.compression_mm = comp_mm;
        result.bm_absolute = extractBiomarkers(dic_abs.poi_queue, comp_mm);

        std::cout << "[BM]   Biomarkers done : "
            << (omp_get_wtime() - bm_t0) << "s\n";

        if (Config::PRINT_BIOMARKERS)
            printBiomarkers(result.bm_absolute, fr.frame_index);

        // ── Chained DIC (rolling reference) ──────────────────────────────────
        if (Config::COMPUTE_CHAINED && fr.rolling_ref_idx != 0) {
            int roll_idx = fr.rolling_ref_idx - 1;
            if (roll_idx >= 0 &&
                roll_idx < (int)prepared_paths.size() &&
                roll_idx != (int)i)
            {
                std::cout << "[MAIN]   Chained DIC: frame " << fr.frame_index
                    << " vs rolling ref frame " << fr.rolling_ref_idx << "\n";

                DICContextGuard chain_guard;
                chain_guard.ctx = createDICContext(prepared_paths[roll_idx]);
                chain_guard.armed = true;

                DICResult dic_chain = runDIC(chain_guard.ctx,
                    prepared_paths[i],
                    fr.frame_index);

                result.bm_chained = extractBiomarkers(dic_chain.poi_queue);
                result.has_chained = true;

                std::cout << "[MAIN]   Chained displacement: "
                    << result.bm_chained.displacement << " px\n";
            }
        }

        std::cout << "[MAIN] Frame " << fr.frame_index
            << " total time : "
            << (omp_get_wtime() - frame_t0) << "s\n";

        all_results.push_back(std::move(result));
    }

    // ── Write results ─────────────────────────────────────────────────────────
    std::cout << "\n[MAIN]   Writing results ...\n";
    writeFrameResults(all_results, output_folder);
    writeSummary(all_results, output_folder);
    writeBehaviourSummary(all_results, output_folder);
    writeDatasetQuality(all_results, output_folder);

    printFinalSummaryFullField(all_results, output_folder);
}


// =============================================================================
// runROIBasedPipeline
// =============================================================================
// Five-ROI ISO-aligned DIC path.

static void runROIBasedPipeline(
    const SessionData& session,
    const std::string& ref_prepared,
    const std::vector<std::string>& prepared_paths,
    const std::string& output_folder,
    int                             img_w,
    int                             img_h)
{
    // ── RAII guard — destroyROIDICContext even on exception ───────────────────
    struct ROIContextGuard {
        ROIDICContext ctx;
        bool          armed = false;
        ~ROIContextGuard() { if (armed) destroyROIDICContext(ctx); }
    };

    std::cout << "[MAIN]   Building ROI DIC contexts (5 ROIs) ...\n";
    ROIContextGuard guard;
    guard.ctx = createROIDICContext(ref_prepared, img_w, img_h);
    guard.armed = true;
    ROIDICContext& roi_ctx = guard.ctx;
    std::cout << "[MAIN]   ROI contexts ready.\n\n";

    // Placement radius — needed for biomarker geometry
    int optic_r = g_calibration.valid
        ? (int)(Config::ROI_RADIUS_MM / g_calibration.pixel_size_mm_per_px)
        : Config::ROI_RADIUS_PX_FALLBACK;
    double placement_r = optic_r * Config::ROI_PLACEMENT_FRACTION;

    std::vector<ROIFrameResult> all_results;
    all_results.reserve(session.frames.size());

    for (size_t i = 0; i < session.frames.size(); ++i) {
        double frame_t0 = omp_get_wtime();
        const FrameRecord& fr = session.frames[i];

        // ── Five-ROI DIC ──────────────────────────────────────────────────────
        ROIDICResult roi_dic = runROIDIC(roi_ctx,
            ref_prepared,
            prepared_paths[i],
            fr.frame_index);
        printROIDICResult(roi_dic, fr.frame_index);

        // ── Biomarkers ────────────────────────────────────────────────────────
        ROIBiomarkers bm = extractROIBiomarkers(roi_dic, placement_r,
            getCompressionMm(session, fr.frame_index));
        if (Config::PRINT_BIOMARKERS)
            printROIBiomarkers(bm, fr.frame_index);

        // ── Assemble result record ────────────────────────────────────────────
        ROIFrameResult rfr;
        rfr.frame_index = fr.frame_index;
        rfr.timestamp = fr.timestamp;
        rfr.compression_mm = getCompressionMm(session, fr.frame_index);
        rfr.dic = std::move(roi_dic);
        rfr.bm = bm;

        std::cout << "[MAIN] Frame " << fr.frame_index
            << " total time : "
            << (omp_get_wtime() - frame_t0) << "s\n";

        all_results.push_back(std::move(rfr));
    }

    // ── Write results ─────────────────────────────────────────────────────────
    std::cout << "\n[MAIN]   Writing ROI results ...\n";
    writeROIFrameResults(all_results, output_folder);
    writeROISummary(all_results, output_folder);
    writeROIBehaviourSummary(all_results, output_folder);
    writeDatasetQuality(all_results, output_folder);

    printFinalSummaryROI(all_results, output_folder);
}


// =============================================================================
// main
// =============================================================================

int main()
{
    printStartBanner(Config::SESSION_FOLDER, Config::ROI_BASED_MODE);

    try {
        // ── Setup ─────────────────────────────────────────────────────────────
        fs::create_directories(Config::OUTPUT_FOLDER);
        loadCalibration();

        SessionData session = loadSession(Config::SESSION_FOLDER);
        loadOptic(Config::SESSION_FOLDER);
        std::string output_folder = buildOutputFolder(session.session_id);

        // ── Read image dimensions from reference ──────────────────────────────
        cv::Mat ref_check = cv::imread(session.ref_path, cv::IMREAD_GRAYSCALE);
        if (ref_check.empty())
            throw std::runtime_error("Cannot load reference: " + session.ref_path);
        int img_h = ref_check.rows;
        int img_w = ref_check.cols;
        ref_check.release();
        std::cout << "[MAIN]   Image size : " << img_w << " x " << img_h << "\n\n";

        // ── Prepare reference image ───────────────────────────────────────────
        std::cout << "[MAIN]   Preparing reference image ...\n";
        std::string ref_prepared = prepareImageFile(
            session.ref_path, Config::OUTPUT_FOLDER);
        std::cout << "[MAIN]   Reference ready : " << ref_prepared << "\n\n";

        // ── Pre-prepare all deformed frames ───────────────────────────────────
        // Both modes need prepared paths — done once here before branching.
        std::cout << "[MAIN]   Preparing all deformed frames ...\n";
        std::vector<std::string> prepared_paths;
        prepared_paths.reserve(session.frames.size());

        for (size_t i = 0; i < session.frames.size(); ++i) {
            prepared_paths.push_back(
                prepareImageFile(session.frames[i].path, Config::OUTPUT_FOLDER));
            std::cout << "[MAIN]   Frame " << std::setw(4) << std::setfill('0')
                << (i + 1) << " / " << session.frames.size()
                << "\r" << std::flush;
        }
        std::cout << "\n[MAIN]   All frames prepared.\n\n";

        // ── Branch on mode ────────────────────────────────────────────────────
        if (Config::ROI_BASED_MODE) {
            std::cout << "[MAIN]   ── ROI-BASED MODE ─────────────────────────\n\n";
            runROIBasedPipeline(session, ref_prepared, prepared_paths,
                output_folder, img_w, img_h);
        }
        else {
            std::cout << "[MAIN]   ── FULL-FIELD MODE ────────────────────────\n\n";
            runFullFieldPipeline(session, ref_prepared, prepared_paths,
                output_folder);
        }

        // ── Cleanup ───────────────────────────────────────────────────────────
        // Both modes write to the same prepared/ folder.
        // Delete after all DIC contexts are destroyed (RAII guards have exited).
        cleanupPreparedImages(output_folder);
    }
    catch (const std::exception& ex) {
        std::cerr << "\n[FATAL] " << ex.what() << "\n";
        return 1;
    }

    std::cout << "Press Enter to exit...";
    std::cin.get();
    return 0;
}