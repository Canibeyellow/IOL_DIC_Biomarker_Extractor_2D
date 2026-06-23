// =============================================================================
// biomarkers_roi.cpp
// =============================================================================

#include "biomarkers_roi.h"
#include "calibration_runtime.h"
#include "config.h"
#include <iostream>
#include <iomanip>
#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static double pxToMm(double px) {
    return g_calibration.valid
        ? px * g_calibration.pixel_size_mm_per_px : 0.0;
}

ROIBiomarkers extractROIBiomarkers(const ROIDICResult& result,
    double placement_radius_px,
    double compression_mm)
{
    ROIBiomarkers bm;
    bm.failed_rois = result.failed_roi_count;
    bm.used_interpolation = result.interpolated_roi_count > 0;

    if (!result.frame_valid) {
        bm.valid = false;
        bm.message = "Frame invalid — too many failed ROIs";
        return bm;
    }

    // Convenience references — index order: 0=C, 1=P, 2=Q, 3=R, 4=S
    const auto& C = result.rois[0];
    const auto& P = result.rois[1];
    const auto& Q = result.rois[2];
    const auto& R = result.rois[3];
    const auto& S = result.rois[4];

    // All must be valid at this point (either measured or interpolated)
    if (!C.valid || !P.valid || !Q.valid || !R.valid || !S.valid) {
        bm.valid = false;
        bm.message = "One or more ROIs still invalid after interpolation";
        return bm;
    }

    // ── Achieved lever arms (post-clamp) ──────────────────────────────────────
    // P/R separated along v (vertical); Q/S along u (horizontal).
    // Using the true separation keeps tilt/rotation correct even if a ROI
    // was clamped to fit image bounds.
    const auto& defP = result.defs[1];
    const auto& defQ = result.defs[2];
    const auto& defR = result.defs[3];
    const auto& defS = result.defs[4];

    double pr_sep = std::abs((double)defP.centre_y - (double)defR.centre_y);
    double qs_sep = std::abs((double)defQ.centre_x - (double)defS.centre_x);

    if (pr_sep < 1.0 || qs_sep < 1.0) {
        bm.valid = false;
        bm.message = "Degenerate ROI separation — cannot compute tilt/rotation";
        return bm;
    }

    // ── Decentration (with one-sided compression correction) ──────────────────
    double u_dec = C.u_mean;
    double v_dec = C.v_mean;

    if (Config::SESSION_MODE == Config::SessionMode::COMPRESSION
        && compression_mm > 0.0 && g_calibration.valid)
    {
        double compression_px = compression_mm / g_calibration.pixel_size_mm_per_px;
        double correction_px = compression_px * Config::COMPRESSION_SYMMETRY_FACTOR;
        if (Config::COMPRESSION_AXIS == 'u') u_dec -= correction_px;
        else                                 v_dec -= correction_px;
    }
    else if (Config::SESSION_MODE == Config::SessionMode::COMPRESSION
        && compression_mm > 0.0 && !g_calibration.valid)
    {
        bm.message += " [WARNING: compression correction skipped — no calibration]";
    }

    bm.u_dec_px = u_dec;
    bm.v_dec_px = v_dec;
    bm.dec_px = std::sqrt(u_dec * u_dec + v_dec * v_dec);
    bm.u_dec_mm = pxToMm(bm.u_dec_px);
    bm.v_dec_mm = pxToMm(bm.v_dec_px);
    bm.dec_mm = pxToMm(bm.dec_px);

    // ── Tilt (differential displacement / achieved lever arm) ─────────────────
    // tilt_x (perp to compression): u-gradient across the P-R (vertical) axis
    // tilt_y (along compression):   v-gradient across the Q-S (horizontal) axis
    bm.tilt_x_rad = (R.u_mean - P.u_mean) / pr_sep;
    bm.tilt_y_rad = (Q.v_mean - S.v_mean) / qs_sep;
    bm.tilt_deg = std::atan(std::sqrt(
        bm.tilt_x_rad * bm.tilt_x_rad +
        bm.tilt_y_rad * bm.tilt_y_rad))
        * 180.0 / M_PI;

    // ── Rotation (rigid-body in-plane angle, averaged over both axis pairs) ────
    // Q-S pair: ref vector along +u with length qs_sep.
    //   ref = (qs_sep, 0),  def = (qs_sep + Q.u - S.u,  Q.v - S.v)
    // P-R pair: ref vector along -v with length pr_sep (P above R; in image
    //   coords R is at larger y, so R->P points in -v).
    //   ref = (0, -pr_sep),  def = (P.u - R.u,  -pr_sep + (P.v - R.v))
    auto angleFrom = [](double ref_x, double ref_y,
        double def_x, double def_y) -> double {
            double cross = ref_x * def_y - ref_y * def_x;
            double dot = ref_x * def_x + ref_y * def_y;
            return -std::atan2(cross, dot) * 180.0 / M_PI;
        };

    double qs_rot = angleFrom(qs_sep, 0.0,
        qs_sep + (Q.u_mean - S.u_mean),
        Q.v_mean - S.v_mean);

    double pr_rot = angleFrom(0.0, -pr_sep,
        P.u_mean - R.u_mean,
        -pr_sep + (P.v_mean - R.v_mean));

    bool qs_interp = Q.interpolated || S.interpolated;
    bool pr_interp = P.interpolated || R.interpolated;

    if (!qs_interp && !pr_interp) {
        bm.rotation_deg = (qs_rot + pr_rot) / 2.0;
    }
    else if (!qs_interp) {
        bm.rotation_deg = qs_rot;
        bm.message += " [rotation from Q-S only: P or R interpolated]";
    }
    else if (!pr_interp) {
        bm.rotation_deg = pr_rot;
        bm.message += " [rotation from P-R only: Q or S interpolated]";
    }
    else {
        bm.rotation_deg = (qs_rot + pr_rot) / 2.0;
        bm.message += " [rotation from partially interpolated pairs]";
    }

    // ── Finalise: status first, then any accumulated notes ────────────────────
    bm.valid = true;
    std::string status = bm.used_interpolation
        ? "valid (interpolation used)" : "valid";
    bm.message = status + bm.message;
    return bm;
}


void printROIBiomarkers(const ROIBiomarkers& bm, int frame_index)
{
    std::cout << "\n[ROI-BM]  Frame " << frame_index << " Biomarkers\n";
    if (!bm.valid) {
        std::cout << "[ROI-BM]  INVALID: " << bm.message << "\n";
        return;
    }
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "[ROI-BM]  Decentration  : u=" << bm.u_dec_px
        << "px  v=" << bm.v_dec_px << "px"
        << "  mag=" << bm.dec_px << "px";
    if (g_calibration.valid)
        std::cout << "  (" << bm.dec_mm << "mm)";
    std::cout << "\n";
    std::cout << "[ROI-BM]  Tilt          : " << bm.tilt_deg << " deg"
        << "  (t_perp_compression=" << bm.tilt_x_rad * 180.0 / M_PI << "deg"
        << "  t_along_compression=" << bm.tilt_y_rad * 180.0 / M_PI << "deg)\n";
    std::cout << "[ROI-BM]  Rotation      : " << bm.rotation_deg << " deg\n";
    if (!bm.message.empty() && bm.message != "valid")
        std::cout << "[ROI-BM]  Note          : " << bm.message << "\n";
    if (bm.used_interpolation)
        std::cout << "[ROI-BM]  NOTE: " << bm.failed_rois
        << " ROI(s) were interpolated\n";
}