// =============================================================================
// results_writer_roi.cpp
// =============================================================================

#include "results_writer_roi.h"
#include "calibration_runtime.h"
#include "config.h"
#include <fstream>
#include <iostream>
#include <filesystem>
#include <iomanip>
#include <cmath>
#include <numeric>
#include <algorithm>
#include "optic_runtime.h"
#include <opencv2/opencv.hpp>
#include <cstdio>
namespace fs = std::filesystem;

static double pxToMm(double px) {
    return g_calibration.valid
        ? px * g_calibration.pixel_size_mm_per_px : 0.0;
}

void writeROIFrameResults(const std::vector<ROIFrameResult>& results,
    const std::string& output_folder)
{
    fs::create_directories(output_folder);
    std::string path = output_folder + "/roi_frame_results.csv";
    std::ofstream f(path);
    if (!f.is_open())
        throw std::runtime_error("Cannot write: " + path);

    f << std::fixed << std::setprecision(6);

    // Header
    f << "frame_index,timestamp,compression_mm,"
        // Global shift
        << "gs_u_px,gs_v_px,gs_zncc,"
        // Per-ROI displacements
        << "C_u_px,C_v_px,C_zncc,C_valid,C_interp,"
        << "P_u_px,P_v_px,P_zncc,P_valid,P_interp,"
        << "Q_u_px,Q_v_px,Q_zncc,Q_valid,Q_interp,"
        << "R_u_px,R_v_px,R_zncc,R_valid,R_interp,"
        << "S_u_px,S_v_px,S_zncc,S_valid,S_interp,"
        // Biomarkers px
        << "u_dec_px,v_dec_px,dec_px,"
        // Biomarkers mm
        << "u_dec_mm,v_dec_mm,dec_mm,"
        // Angular
        << "tilt_deg,rotation_deg,"
        // Quality
        << "failed_rois,used_interpolation,frame_valid,bm_valid\n";

    for (const auto& r : results) {
        f << r.frame_index << ","
            << r.timestamp << ","
            << r.compression_mm << ","
            << r.dic.global_shift_u << ","
            << r.dic.global_shift_v << ","
            << r.dic.global_shift_response << ",";

        for (const auto& roi : r.dic.rois) {
            f << roi.u_mean << ","
                << roi.v_mean << ","
                << roi.mean_zncc << ","
                << (roi.valid ? "1" : "0") << ","
                << (roi.interpolated ? "1" : "0") << ",";
        }

        f << r.bm.u_dec_px << ","
            << r.bm.v_dec_px << ","
            << r.bm.dec_px << ","
            << r.bm.u_dec_mm << ","
            << r.bm.v_dec_mm << ","
            << r.bm.dec_mm << ","
            << r.bm.tilt_deg << ","
            << r.bm.rotation_deg << ","
            << r.dic.failed_roi_count << ","
            << (r.dic.interpolated_roi_count > 0 ? "1" : "0") << ","
            << (r.dic.frame_valid ? "1" : "0") << ","
            << (r.bm.valid ? "1" : "0") << "\n";
    }

    f.close();
    std::cout << "[ROI-RESULTS] roi_frame_results.csv ? " << path << "\n";
}

void writeROISummary(const std::vector<ROIFrameResult>& results,
    const std::string& output_folder)
{
    fs::create_directories(output_folder);
    std::string path = output_folder + "/roi_summary.csv";
    std::ofstream f(path);
    if (!f.is_open())
        throw std::runtime_error("Cannot write: " + path);

    f << std::fixed << std::setprecision(6);

    // Valid-frame series, paired with compression
    std::vector<double> comp, dec, tilt, rot;
    int n_frame_valid = 0, n_bm_valid = 0, n_interp = 0;
    for (const auto& r : results) {
        if (r.dic.frame_valid) ++n_frame_valid;
        if (r.dic.interpolated_roi_count > 0) ++n_interp;
        if (!r.bm.valid) continue;
        ++n_bm_valid;
        comp.push_back(r.compression_mm);
        dec.push_back(r.bm.dec_mm);
        tilt.push_back(r.bm.tilt_deg);
        rot.push_back(r.bm.rotation_deg);
    }

    auto meanv = [](const std::vector<double>& v) {
        return v.empty() ? 0.0 : std::accumulate(v.begin(), v.end(), 0.0) / v.size(); };
    auto minv = [](const std::vector<double>& v) {
        return v.empty() ? 0.0 : *std::min_element(v.begin(), v.end()); };
    auto maxv = [](const std::vector<double>& v) {
        return v.empty() ? 0.0 : *std::max_element(v.begin(), v.end()); };
    auto stdv = [&](const std::vector<double>& v) {
        if (v.size() < 2) return 0.0;
        double m = meanv(v), s = 0.0; for (double x : v) s += (x - m) * (x - m);
        return std::sqrt(s / v.size()); };
    // least-squares slope of y vs compression; ok=false if compression is flat
    auto slope = [&](const std::vector<double>& y, bool& ok) {
        ok = false;
        if (comp.size() < 2) return 0.0;
        double mx = meanv(comp), my = meanv(y), sxx = 0.0, sxy = 0.0;
        for (size_t i = 0; i < comp.size(); ++i) {
            sxx += (comp[i] - mx) * (comp[i] - mx);
            sxy += (comp[i] - mx) * (y[i] - my);
        }
        if (sxx < 1e-9) return 0.0;
        ok = true; return sxy / sxx; };

    // Value at maximum compression (last bm-valid frame at highest compression)
    int idx_max = -1; double best = -1e30;
    for (size_t i = 0; i < results.size(); ++i) {
        if (!results[i].bm.valid) continue;
        if (results[i].compression_mm >= best) { best = results[i].compression_mm; idx_max = (int)i; }
    }
    double dec_max = idx_max >= 0 ? results[idx_max].bm.dec_mm : 0.0;
    double tilt_max = idx_max >= 0 ? results[idx_max].bm.tilt_deg : 0.0;
    double rot_max = idx_max >= 0 ? results[idx_max].bm.rotation_deg : 0.0;

    // --- Dataset block ---
    f << "section,key,value\n";
    f << "dataset,frames_total," << results.size() << "\n";
    f << "dataset,frames_frame_valid," << n_frame_valid << "\n";
    f << "dataset,frames_bm_valid," << n_bm_valid << "\n";
    f << "dataset,frames_with_interpolation," << n_interp << "\n";
    f << "dataset,compression_min_mm," << minv(comp) << "\n";
    f << "dataset,compression_max_mm," << maxv(comp) << "\n";
    f << "dataset,pixel_size_mm_per_px,"
        << (g_calibration.valid ? g_calibration.pixel_size_mm_per_px : 0.0) << "\n";

    // --- Biomarker behaviour (comparison-ready) ---
    bool okd, okt, okr;
    double sl_d = slope(dec, okd), sl_t = slope(tilt, okt), sl_r = slope(rot, okr);
    f << "\nbiomarker,min,max,mean,std,at_max_compression,slope_per_mm,slope_valid\n";
    f << "decentration_mm," << minv(dec) << "," << maxv(dec) << "," << meanv(dec) << ","
        << stdv(dec) << "," << dec_max << "," << sl_d << "," << (okd ? 1 : 0) << "\n";
    f << "tilt_deg," << minv(tilt) << "," << maxv(tilt) << "," << meanv(tilt) << ","
        << stdv(tilt) << "," << tilt_max << "," << sl_t << "," << (okt ? 1 : 0) << "\n";
    f << "rotation_deg," << minv(rot) << "," << maxv(rot) << "," << meanv(rot) << ","
        << stdv(rot) << "," << rot_max << "," << sl_r << "," << (okr ? 1 : 0) << "\n";

    // --- Per-ROI tracking reliability ---
    f << "\nroi,valid_frames,total_frames,valid_fraction,mean_zncc\n";
    int nf = (int)results.size();
    for (int i = 0; i < 5; ++i) {
        int vcount = 0; double zsum = 0.0; std::string label;
        for (const auto& r : results) {
            if (r.dic.rois[i].valid) ++vcount;
            zsum += r.dic.rois[i].mean_zncc;
            if (label.empty() && !r.dic.defs[i].label.empty()) label = r.dic.defs[i].label;
        }
        if (label.empty()) label = std::string(1, "CPQRS"[i]);
        f << label << "," << vcount << "," << nf << ","
            << (nf ? (double)vcount / nf : 0.0) << ","
            << (nf ? zsum / nf : 0.0) << "\n";
    }

    f.close();
    std::cout << "[ROI-RESULTS] roi_summary.csv -> " << path << "\n";
}

// Draws optic circle + 5 ROI boxes (+ optional displacement arrows) onto vis.
static void drawROIScene(cv::Mat& vis, const ROIDICResult& dic,
    double comp_mm, bool show_disp, int gain)
{
    EffectiveROI er = resolveEffectiveROI();
    cv::circle(vis, cv::Point(er.cx, er.cy), er.radius_px, cv::Scalar(255, 130, 59), 3);
    cv::drawMarker(vis, cv::Point(er.cx, er.cy), cv::Scalar(255, 130, 59),
        cv::MARKER_CROSS, 40, 3);

    for (int i = 0; i < 5; ++i) {
        const ROIDefinition& d = dic.defs[i];
        const ROIPointResult& p = dic.rois[i];
        int hs = d.half_size;
        cv::Scalar col = p.valid ? cv::Scalar(0, 255, 0)
            : (p.interpolated ? cv::Scalar(0, 180, 255)
                : cv::Scalar(0, 0, 255));
        cv::rectangle(vis, cv::Point(d.centre_x - hs, d.centre_y - hs),
            cv::Point(d.centre_x + hs, d.centre_y + hs), col, 3);

        std::string lab = d.label.empty() ? std::string(1, "CPQRS"[i]) : d.label;
        char tag[64];
        std::snprintf(tag, sizeof(tag), "%s z=%.2f", lab.c_str(), p.mean_zncc);
        cv::putText(vis, tag, cv::Point(d.centre_x - hs, d.centre_y - hs - 12),
            cv::FONT_HERSHEY_SIMPLEX, 1.2, col, 3, cv::LINE_AA);

        if (show_disp && p.valid) {
            cv::Point a(d.centre_x, d.centre_y);
            cv::Point b(d.centre_x + (int)std::lround(p.u_mean * gain),
                d.centre_y + (int)std::lround(p.v_mean * gain));
            cv::arrowedLine(vis, a, b, cv::Scalar(0, 255, 255), 3, cv::LINE_AA, 0, 0.3);
        }
    }

    char hud[128];
    std::snprintf(hud, sizeof(hud), "compression=%.3f mm   disp_gain=x%d", comp_mm, gain);
    cv::putText(vis, hud, cv::Point(30, 50), cv::FONT_HERSHEY_SIMPLEX,
        1.3, cv::Scalar(255, 255, 255), 3, cv::LINE_AA);
}

void drawROIOverlays(const std::vector<ROIFrameResult>& results,
    const SessionData& session,
    const std::string& output_folder)
{
    if (!Config::ROI_DRAW_OVERLAYS) return;
    fs::path dir = fs::path(output_folder) / "roi_overlays";
    fs::create_directories(dir);
    int gain = Config::ROI_OVERLAY_DISP_GAIN;

    // Reference frame: placement only (no displacement)
    if (!results.empty() && !session.ref_path.empty()) {
        cv::Mat ref = cv::imread(session.ref_path, cv::IMREAD_GRAYSCALE);
        if (!ref.empty()) {
            cv::Mat vis; cv::cvtColor(ref, vis, cv::COLOR_GRAY2BGR);
            drawROIScene(vis, results.front().dic, 0.0, /*show_disp=*/false, gain);
            cv::imwrite((dir / "roi_overlay_REF.png").string(), vis);
        }
    }

    // Each deformed frame: boxes + displacement arrows
    for (const auto& r : results) {
        std::string img_path;
        for (const auto& fr : session.frames)
            if (fr.frame_index == r.frame_index) { img_path = fr.path; break; }
        if (img_path.empty()) continue;
        cv::Mat img = cv::imread(img_path, cv::IMREAD_GRAYSCALE);
        if (img.empty()) {
            std::cout << "[ROI-OVERLAY] skip (cannot read) " << img_path << "\n";
            continue;
        }
        cv::Mat vis; cv::cvtColor(img, vis, cv::COLOR_GRAY2BGR);
        drawROIScene(vis, r.dic, r.compression_mm, /*show_disp=*/true, gain);
        char name[64];
        std::snprintf(name, sizeof(name), "roi_overlay_frame_%04d.png", r.frame_index);
        cv::imwrite((dir / name).string(), vis);
    }
    std::cout << "[ROI-RESULTS] ROI overlays -> " << dir.string() << "\n";
}

void writeROIBehaviourSummary(const std::vector<ROIFrameResult>& results,
    const std::string& output_folder)
{
    fs::create_directories(output_folder);
    std::string path = output_folder + "/roi_behaviour_summary.txt";
    std::ofstream f(path);
    if (!f.is_open())
        throw std::runtime_error("Cannot write: " + path);

    // --- gather valid-frame series, paired with compression ---
    std::vector<double> comp, dec, tilt, rot;
    int n_frame_valid = 0, n_bm_valid = 0, n_interp = 0;
    for (const auto& r : results) {
        if (r.dic.frame_valid) ++n_frame_valid;
        if (r.dic.interpolated_roi_count > 0) ++n_interp;
        if (!r.bm.valid) continue;
        ++n_bm_valid;
        comp.push_back(r.compression_mm);
        dec.push_back(r.bm.dec_mm);
        tilt.push_back(r.bm.tilt_deg);
        rot.push_back(r.bm.rotation_deg);
    }

    auto meanv = [&](const std::vector<double>& v) {
        return v.empty() ? 0.0 : std::accumulate(v.begin(), v.end(), 0.0) / v.size(); };
    auto minv = [&](const std::vector<double>& v) {
        return v.empty() ? 0.0 : *std::min_element(v.begin(), v.end()); };
    auto maxv = [&](const std::vector<double>& v) {
        return v.empty() ? 0.0 : *std::max_element(v.begin(), v.end()); };
    auto stdv = [&](const std::vector<double>& v) {
        if (v.size() < 2) return 0.0;
        double m = meanv(v), s = 0.0; for (double x : v) s += (x - m) * (x - m);
        return std::sqrt(s / v.size()); };
    auto slope = [&](const std::vector<double>& y, bool& ok) {
        ok = false;
        if (comp.size() < 2) return 0.0;
        double mx = meanv(comp), my = meanv(y), sxx = 0.0, sxy = 0.0;
        for (size_t i = 0; i < comp.size(); ++i) {
            sxx += (comp[i] - mx) * (comp[i] - mx);
            sxy += (comp[i] - mx) * (y[i] - my);
        }
        if (sxx < 1e-9) return 0.0;
        ok = true; return sxy / sxx; };

    int idx_max = -1; double best = -1e30;
    for (size_t i = 0; i < results.size(); ++i) {
        if (!results[i].bm.valid) continue;
        if (results[i].compression_mm >= best) { best = results[i].compression_mm; idx_max = (int)i; }
    }

    std::string date = results.empty() ? "(none)" : results.front().timestamp;

    // --- header / dataset ---
    f << "===============================================================\n";
    f << "  IOL BEHAVIOUR SUMMARY\n";
    f << "===============================================================\n";
    f << "Date (first frame) : " << date << "\n";
    f << "Frames total       : " << results.size() << "\n";
    f << "Frames frame-valid : " << n_frame_valid << "\n";
    f << "Frames bm-valid    : " << n_bm_valid << "\n";
    f << "Frames interpolated: " << n_interp << "\n";
    f << std::fixed << std::setprecision(3);
    f << "Compression range  : " << minv(comp) << " -> " << maxv(comp) << " mm\n";
    f << std::setprecision(6);
    f << "Pixel size         : "
        << (g_calibration.valid ? g_calibration.pixel_size_mm_per_px : 0.0)
        << " mm/px" << (g_calibration.valid ? "" : "  (UNCALIBRATED)") << "\n\n";

    // --- per-ROI tracking reliability ---
    f << "ROI TRACKING RELIABILITY\n";
    f << "---------------------------------------------------------------\n";
    f << std::left << std::setw(6) << "ROI"
        << std::right << std::setw(14) << "valid/total"
        << std::setw(14) << "valid_frac"
        << std::setw(14) << "mean_zncc" << "\n";
    int nf = (int)results.size();
    for (int i = 0; i < 5; ++i) {
        int vc = 0; double zs = 0.0; std::string lab;
        for (const auto& r : results) {
            if (r.dic.rois[i].valid) ++vc;
            zs += r.dic.rois[i].mean_zncc;
            if (lab.empty() && !r.dic.defs[i].label.empty()) lab = r.dic.defs[i].label;
        }
        if (lab.empty()) lab = std::string(1, "CPQRS"[i]);
        std::string vt = std::to_string(vc) + "/" + std::to_string(nf);
        f << std::left << std::setw(6) << lab
            << std::right << std::setw(14) << vt
            << std::setw(14) << std::setprecision(3) << (nf ? (double)vc / nf : 0.0)
            << std::setw(14) << (nf ? zs / nf : 0.0) << "\n";
    }
    f << "\n";

    // --- biomarker behaviour ---
    bool okd, okt, okr;
    double sld = slope(dec, okd), slt = slope(tilt, okt), slr = slope(rot, okr);
    auto row = [&](const std::string& name, const std::vector<double>& v,
        double atmax, double sl, bool ok, const std::string& note) {
            f << std::left << std::setw(16) << name
                << std::right << std::fixed << std::setprecision(4)
                << std::setw(11) << minv(v) << std::setw(11) << maxv(v)
                << std::setw(11) << meanv(v) << std::setw(11) << stdv(v)
                << std::setw(13) << atmax;
            if (ok) f << std::setw(13) << sl;
            else    f << std::setw(13) << "N/A";
            f << "   " << note << "\n";
        };
    f << "BIOMARKER BEHAVIOUR  (valid frames only)\n";
    f << "---------------------------------------------------------------\n";
    f << std::left << std::setw(16) << "biomarker"
        << std::right << std::setw(11) << "min" << std::setw(11) << "max"
        << std::setw(11) << "mean" << std::setw(11) << "std"
        << std::setw(13) << "at_max_comp" << std::setw(13) << "slope/mm" << "\n";
    row("decentration_mm", dec, idx_max >= 0 ? results[idx_max].bm.dec_mm : 0.0, sld, okd, "");
    row("tilt_deg", tilt, idx_max >= 0 ? results[idx_max].bm.tilt_deg : 0.0, slt, okt, "in-plane proxy");
    row("rotation_deg", rot, idx_max >= 0 ? results[idx_max].bm.rotation_deg : 0.0, slr, okr, "+ = CCW");
    f << "\n";

    // --- final state ---
    f << "FINAL STATE (frame at max compression)\n";
    f << "---------------------------------------------------------------\n";
    if (idx_max >= 0) {
        const auto& b = results[idx_max].bm;
        f << std::setprecision(3);
        f << "  compression  : " << results[idx_max].compression_mm << " mm\n";
        f << std::setprecision(4);
        f << "  decentration : " << b.dec_mm << " mm   (u=" << b.u_dec_mm
            << ", v=" << b.v_dec_mm << ")\n";
        f << "  tilt         : " << b.tilt_deg << " deg\n";
        f << "  rotation     : " << b.rotation_deg << " deg\n";
    }
    else {
        f << "  (no biomarker-valid frame)\n";
    }
    f << "\nNotes: decentration = |u,v| of the centre ROI; tilt is a 2D in-plane\n";
    f << "proxy; rotation sign positive = counter-clockwise. 'slope/mm' is the\n";
    f << "least-squares rate vs compression (N/A when compression is constant).\n";

    f.close();
    std::cout << "[ROI-RESULTS] roi_behaviour_summary.txt -> " << path << "\n";
}

void writeDatasetQuality(const std::vector<ROIFrameResult>& results,
    const std::string& output_folder)
{
    fs::create_directories(output_folder);
    std::string path = output_folder + "/dataset_quality.txt";
    std::ofstream f(path);
    if (!f.is_open())
        throw std::runtime_error("Cannot write: " + path);

    const int nf = (int)results.size();

    int n_frame_valid = 0, n_bm_valid = 0, n_interp_frames = 0, n_failed_frames = 0;
    std::array<int, 5>    roi_valid{};
    std::array<double, 5> roi_zncc_sum{};
    std::array<double, 5> roi_zncc_min;  roi_zncc_min.fill(1e30);
    std::array<double, 5> roi_vf_sum{};
    std::array<std::string, 5> roi_label;
    std::map<std::string, int> fail_tally;
    std::vector<double> gs_resp, frame_time;

    for (const auto& r : results) {
        if (r.dic.frame_valid) ++n_frame_valid; else ++n_failed_frames;
        if (r.bm.valid) ++n_bm_valid;
        if (r.dic.interpolated_roi_count > 0) ++n_interp_frames;
        gs_resp.push_back(r.dic.global_shift_response);
        frame_time.push_back(r.dic.time_total_sec);
        for (int i = 0; i < 5; ++i) {
            const auto& p = r.dic.rois[i];
            if (roi_label[i].empty())
                roi_label[i] = r.dic.defs[i].label.empty()
                ? std::string(1, "CPQRS"[i]) : r.dic.defs[i].label;
            if (p.valid) ++roi_valid[i];
            roi_zncc_sum[i] += p.mean_zncc;
            roi_zncc_min[i] = std::min(roi_zncc_min[i], p.mean_zncc);
            roi_vf_sum[i] += p.valid_fraction;
            if (!p.valid && !p.fail_reason.empty())
                fail_tally[p.fail_reason]++;
        }
    }

    auto mean = [](const std::vector<double>& v) {
        return v.empty() ? 0.0 : std::accumulate(v.begin(), v.end(), 0.0) / v.size(); };
    auto vmin = [](const std::vector<double>& v) {
        return v.empty() ? 0.0 : *std::min_element(v.begin(), v.end()); };

    double frame_valid_frac = nf ? (double)n_frame_valid / nf : 0.0;
    double overall_zncc = 0.0;
    for (int i = 0; i < 5; ++i) overall_zncc += nf ? roi_zncc_sum[i] / nf : 0.0;
    overall_zncc /= 5.0;
    double worst_roi_frac = 1.0;
    for (int i = 0; i < 5; ++i)
        worst_roi_frac = std::min(worst_roi_frac, nf ? (double)roi_valid[i] / nf : 0.0);

    const char* tier = "POOR";
    if (frame_valid_frac >= Config::QUALITY_GOOD_FRAME_VALID_FRAC &&
        overall_zncc >= Config::QUALITY_GOOD_MEAN_ZNCC)            tier = "GOOD";
    else if (frame_valid_frac >= Config::QUALITY_OKAY_FRAME_VALID_FRAC &&
        overall_zncc >= Config::QUALITY_OKAY_MEAN_ZNCC)       tier = "OKAY";

    f << std::fixed;
    f << "===============================================================\n";
    f << "  DATASET QUALITY & RELIABILITY\n";
    f << "===============================================================\n";
    f << "Frames total           : " << nf << "\n";
    f << "Frame-valid            : " << n_frame_valid << "  ("
        << std::setprecision(1) << 100.0 * frame_valid_frac << "%)\n";
    f << "Frame-failed           : " << n_failed_frames << "\n";
    f << "Biomarker-valid        : " << n_bm_valid << "\n";
    f << "Frames w/ interpolation: " << n_interp_frames << "\n\n";

    f << "PER-ROI RELIABILITY\n";
    f << "---------------------------------------------------------------\n";
    f << std::left << std::setw(6) << "ROI"
        << std::right << std::setw(13) << "valid/total"
        << std::setw(12) << "valid_frac" << std::setw(12) << "mean_zncc"
        << std::setw(12) << "min_zncc" << std::setw(14) << "mean_poi_vf" << "\n";
    for (int i = 0; i < 5; ++i) {
        std::string vt = std::to_string(roi_valid[i]) + "/" + std::to_string(nf);
        f << std::left << std::setw(6) << roi_label[i]
            << std::right << std::setprecision(3)
            << std::setw(13) << vt
            << std::setw(12) << (nf ? (double)roi_valid[i] / nf : 0.0)
            << std::setw(12) << (nf ? roi_zncc_sum[i] / nf : 0.0)
            << std::setw(12) << (nf ? roi_zncc_min[i] : 0.0)
            << std::setw(14) << (nf ? roi_vf_sum[i] / nf : 0.0) << "\n";
    }
    f << "\n";

    f << "GLOBAL-SHIFT PRE-ALIGNMENT\n";
    f << "---------------------------------------------------------------\n";
    f << std::setprecision(3);
    f << "  response  min : " << vmin(gs_resp) << "\n";
    f << "  response mean : " << mean(gs_resp) << "\n\n";

    f << "FAILURE REASONS (across all ROI failures)\n";
    f << "---------------------------------------------------------------\n";
    if (fail_tally.empty()) f << "  (none)\n";
    else for (const auto& kv : fail_tally)
        f << "  " << std::setw(4) << kv.second << "x  " << kv.first << "\n";
    f << "\n";

    f << "TIMING\n";
    f << "---------------------------------------------------------------\n";
    f << "  total      : " << std::accumulate(frame_time.begin(), frame_time.end(), 0.0) << " s\n";
    f << "  mean/frame : " << mean(frame_time) << " s\n\n";

    f << "RATING INPUTS  (you decide; thresholds in config.h)\n";
    f << "---------------------------------------------------------------\n";
    f << "  frame-valid fraction : " << frame_valid_frac << "\n";
    f << "  overall mean ZNCC    : " << overall_zncc << "\n";
    f << "  worst-ROI valid frac : " << worst_roi_frac << "\n";
    f << "  suggested tier       : " << tier << "   (heuristic, not a verdict)\n";

    f.close();
    std::cout << "[ROI-RESULTS] dataset_quality.txt -> " << path << "\n";
}