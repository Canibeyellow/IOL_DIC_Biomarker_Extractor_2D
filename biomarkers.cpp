// =============================================================================
// biomarkers.cpp
// =============================================================================

#include "biomarkers.h"
#include "config.h"
#include <iostream>
#include <iomanip>
#include <numeric>
#include <cmath>
#include "calibration_runtime.h"
#include "compression_model.h"
#include "dic_pipeline.h"

using namespace opencorr;

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif


// =============================================================================
// extractBiomarkers
// =============================================================================
static double getROIRadiusPxForBiomarkers()
{
    if (Config::ROI_USE_PHYSICAL_SIZE && g_calibration.valid)
        return Config::ROI_RADIUS_MM / g_calibration.pixel_size_mm_per_px;

    return Config::ROI_RADIUS_PX_FALLBACK;
}
Biomarkers extractBiomarkers(const std::vector<POI2D>& poi_queue,
    double compression_mm)
{
    Biomarkers bm;

    // ── Collect valid POIs ────────────────────────────────────────────────────
    // "Valid" = ZNCC above threshold = reliable correlation
    std::vector<const POI2D*> valid;
    valid.reserve(poi_queue.size());

    for (const auto& poi : poi_queue) {
        if (isValidPOI(poi))
            valid.push_back(&poi);
    }

    bm.valid_poi_count = static_cast<int>(valid.size());

    if (bm.valid_poi_count < Config::MIN_POIS_FOR_PLANE) {
        bm.valid   = false;
        bm.message = "Too few valid POIs (" +
                     std::to_string(bm.valid_poi_count) +
                     " < " + std::to_string(Config::MIN_POIS_FOR_PLANE) +
                     "). Check speckle quality and ROI position.";
        return bm;
    }

    // ── Mean ZNCC ─────────────────────────────────────────────────────────────
    double zncc_sum = 0.0;
    for (const auto* p : valid) zncc_sum += p->result.zncc;
    bm.mean_zncc = zncc_sum / valid.size();

    // =========================================================================
    // BIOMARKER 1 & 2 — Displacement and Decentration
    // =========================================================================
    // The mean displacement of all valid POIs = rigid-body translation of the
    // IOL as a whole. This is the centroid shift.
    //
    // In 2D DIC: u = horizontal displacement, v = vertical displacement.
    // Both are in pixels. Convert to mm using pixelsToMm() if pixel pitch known.
    //
    // Decentration = lateral shift of the optical centre = same as mean (u,v)
    // in 2D DIC. In stereo, it will be the XY component of the 3D centroid shift.

    double u_sum = 0.0, v_sum = 0.0;
    for (const auto* p : valid) {
        u_sum += p->deformation.u;
        v_sum += p->deformation.v;
    }

    bm.u_mean      = u_sum / valid.size();
    bm.v_mean      = v_sum / valid.size();
    bm.displacement = std::sqrt(bm.u_mean * bm.u_mean + bm.v_mean * bm.v_mean);
    bm.decentration = bm.displacement;   // identical in 2D DIC
    // ── Compression correction ────────────────────────────────────────────────
    // Store the applied compression for the results writer
    bm.compression_mm = compression_mm;

    if (Config::SESSION_MODE == Config::SessionMode::COMPRESSION
        && compression_mm > 0.0 && g_calibration.valid)
    {
        // Convert compression from mm to pixels
        double compression_px = compression_mm / g_calibration.pixel_size_mm_per_px;

        // One-sided loading: optic centre shifts by compression_px * SYMMETRY_FACTOR
        // along the compression axis. Subtract to isolate true optical decentration.
        double correction_px = compression_px * Config::COMPRESSION_SYMMETRY_FACTOR;

        if (Config::COMPRESSION_AXIS == 'u') {
            bm.u_mean_corrected = bm.u_mean - correction_px;
            bm.v_mean_corrected = bm.v_mean;
        }
        else {
            bm.u_mean_corrected = bm.u_mean;
            bm.v_mean_corrected = bm.v_mean - correction_px;
        }
    }
    else {
        // TRANSLATION mode or no compression data — no correction needed
        bm.u_mean_corrected = bm.u_mean;
        bm.v_mean_corrected = bm.v_mean;

        // ADDED: warn if we skipped correction because calibration is missing
        if (Config::SESSION_MODE == Config::SessionMode::COMPRESSION
            && compression_mm > 0.0
            && !g_calibration.valid)
        {
            bm.message += " [WARNING: compression correction skipped — "
                "calibration not loaded. Decentration includes fixture motion.]";
        }
    }

    // =========================================================================
        // BIOMARKER 3 — Tilt (2D approximation via least-squares plane fit)
        // =========================================================================
        // Fits the plane  u(x, y) = a*x + b*y + c  to the u-displacement field
        // using the normal equations. The gradient magnitude sqrt(a²+b²) is the
        // rate of change of in-plane u with position — a proxy for out-of-plane tilt.
        // Same fit is done for v; the combined gradient is reported.
        //
        // This is an approximation. Stereo DIC gives the true 3D tilt angle.

    {
        double sx = 0, sy = 0, sx2 = 0, sy2 = 0, sxy = 0;
        double su = 0, sv = 0, sxu = 0, syu = 0, sxv = 0, syv = 0;
        double N = (double)valid.size();

        for (const auto* p : valid) {
            double x = p->x, y = p->y;
            double pu = p->deformation.u, pv = p->deformation.v;
            sx += x;   sy += y;
            sx2 += x * x; sy2 += y * y; sxy += x * y;
            su += pu;  sv += pv;
            sxu += x * pu; syu += y * pu;
            sxv += x * pv; syv += y * pv;
        }

        // Normal equations: [sx2 sxy sx] [a]   [sxu]
        //                   [sxy sy2 sy] [b] = [syu]
        //                   [sx  sy  N ] [c]   [su ]
        // Solve 3x3 system via Cramer's rule (small fixed size, no library needed)
        auto solve3 = [](double A[3][3], double b[3], double x[3]) -> bool {
            double det = A[0][0] * (A[1][1] * A[2][2] - A[1][2] * A[2][1])
                - A[0][1] * (A[1][0] * A[2][2] - A[1][2] * A[2][0])
                + A[0][2] * (A[1][0] * A[2][1] - A[1][1] * A[2][0]);
            if (std::abs(det) < 1e-12) return false;
            double inv = 1.0 / det;
            x[0] = inv * ((A[1][1] * A[2][2] - A[1][2] * A[2][1]) * b[0]
                - (A[0][1] * A[2][2] - A[0][2] * A[2][1]) * b[1]
                + (A[0][1] * A[1][2] - A[0][2] * A[1][1]) * b[2]);
            x[1] = inv * (-(A[1][0] * A[2][2] - A[1][2] * A[2][0]) * b[0]
                + (A[0][0] * A[2][2] - A[0][2] * A[2][0]) * b[1]
                - (A[0][0] * A[1][2] - A[0][2] * A[1][0]) * b[2]);
            x[2] = inv * ((A[1][0] * A[2][1] - A[1][1] * A[2][0]) * b[0]
                - (A[0][0] * A[2][1] - A[0][1] * A[2][0]) * b[1]
                + (A[0][0] * A[1][1] - A[0][1] * A[1][0]) * b[2]);
            return true;
            };

        double M[3][3] = { {sx2, sxy, sx}, {sxy, sy2, sy}, {sx, sy, N} };
        double rhu[3] = { sxu, syu, su };
        double rhv[3] = { sxv, syv, sv };
        double cu[3] = { 0,0,0 };
        double cv[3] = { 0,0,0 };

        bool ok_u = solve3(M, rhu, cu);
        bool ok_v = solve3(M, rhv, cv);

        if (ok_u && ok_v) {
            // cu[0]=a, cu[1]=b are the u-gradient components (px/px)
            // cv[0]=a, cv[1]=b are the v-gradient components
            double grad = std::sqrt(cu[0] * cu[0] + cu[1] * cu[1]
                + cv[0] * cv[0] + cv[1] * cv[1]);
            bm.tilt_proxy_deg = std::atan(grad) * (180.0 / M_PI);
        }
        else {
            bm.tilt_proxy_deg = 0.0;
            bm.message += " [Tilt plane fit singular — POIs may be collinear]";
        }
    }

    // =========================================================================
    // BIOMARKER 4 — In-plane Rotation
    // =========================================================================
    // The deformation gradient tensor F has symmetric and antisymmetric parts:
    //   Symmetric part     → strain (stretching / compression)
    //   Antisymmetric part → rigid-body rotation
    //
    // ICGN2D1 outputs the 4 gradient components for each POI:
    //   poi.deformation.ux = du/dx
    //   poi.deformation.uy = du/dy
    //   poi.deformation.vx = dv/dx
    //   poi.deformation.vy = dv/dy
    //
    // In-plane rotation angle:
    //   omega_z = 0.5 * (dv/dx - du/dy)
    // (antisymmetric shear = pure rotation, symmetric shear = shear strain)
    //
    // We average omega_z over all valid POIs for the mean rigid-body rotation.
    // For an IOL rotating as a rigid body, this should be spatially uniform.
    // Spatial variation in omega_z would indicate non-uniform / haptic bending.
    //
    // Units: radians → converted to degrees.
    // Sign convention: positive = counter-clockwise (standard math convention).

    // Use only POIs in the outer annular ring — more sensitive to rotation
    // than the optic centre (which has small lever arm).
    double cx = Config::ROI_CENTER_X;
    double cy = Config::ROI_CENTER_Y;

    double roi_radius_px = getROIRadiusPxForBiomarkers();
    double r_inner = roi_radius_px * Config::ROT_INNER_FRAC;
    double r_outer = roi_radius_px * Config::ROT_OUTER_FRAC;

    double r_inner2 = r_inner * r_inner;
    double r_outer2 = r_outer * r_outer;

    double omega_sum = 0.0;
    int    omega_n   = 0;

    for (const auto* p : valid) {
        double dx = p->x - cx;
        double dy = p->y - cy;
        double r2 = dx*dx + dy*dy;

        if (r2 >= r_inner2 && r2 <= r_outer2) {
            // Antisymmetric shear component = in-plane rotation
            double omega = 0.5 * (p->deformation.vx - p->deformation.uy);
            omega_sum += omega;
            ++omega_n;
        }
    }

    if (omega_n > 0) {
        double omega_mean = omega_sum / omega_n;
        bm.rotation_deg = omega_mean * (180.0 / M_PI);
    } else {
        bm.rotation_deg = 0.0;
        bm.message += " [No POIs in rotation annulus — check ROT_INNER/OUTER_FRAC]";
    }

    bm.valid = true;
    return bm;
}


// =============================================================================
// printBiomarkers
// =============================================================================

void printBiomarkers(const Biomarkers& bm, int frame_index)
{
    std::cout << "\n[BM]  Frame " << std::setw(4) << std::setfill('0')
              << frame_index << " Biomarkers\n";
    std::cout << std::fixed << std::setprecision(4);

    if (!bm.valid) {
        std::cout << "[BM]  INVALID — " << bm.message << "\n";
        return;
    }

    std::cout << "[BM]  Valid POIs    : " << bm.valid_poi_count
              << "  (mean ZNCC=" << bm.mean_zncc << ")\n";
    std::cout << "[BM]  u_mean        : " << bm.u_mean       << " px\n";
    std::cout << "[BM]  v_mean        : " << bm.v_mean       << " px\n";
    std::cout << "[BM]  Displacement  : " << bm.displacement << " px\n";
    std::cout << "[BM]  Decentration  : " << bm.decentration << " px\n";
    if (Config::SESSION_MODE == Config::SessionMode::COMPRESSION) {
        std::cout << "[BM]  Compression   : " << bm.compression_mm << " mm\n";
        std::cout << "[BM]  u_corrected   : " << bm.u_mean_corrected << " px\n";
        std::cout << "[BM]  v_corrected   : " << bm.v_mean_corrected << " px\n";
        std::cout << "[BM]  Decent.(corr) : " << bm.decentration_corrected << " px\n";
    }
    std::cout << "[BM]  Tilt Proximation : " << bm.tilt_proxy_deg     << " deg\n";
    std::cout << "[BM]  Rotation      : " << bm.rotation_deg << " deg\n";

    if (!bm.message.empty())
        std::cout << "[BM]  Note          : " << bm.message << "\n";
}


// =============================================================================
// pixelsToMm
// =============================================================================

double pixelsToMm(double pixels, double pixel_pitch_mm_per_px)
{
    return pixels * pixel_pitch_mm_per_px;
}
