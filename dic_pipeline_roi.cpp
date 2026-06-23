// =============================================================================
// dic_pipeline_roi.cpp
// =============================================================================

#include "dic_pipeline_roi.h"
#include "config.h"
#include "calibration_runtime.h"
#include <iostream>
#include <iomanip>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <omp.h>
#include <opencv2/opencv.hpp>
#include <limits>
#include <array>
#include "optic_runtime.h"

using namespace opencorr;
static double nowSec() { return omp_get_wtime(); }
// =============================================================================
// pyramidTemplateMatch  (internal)
// =============================================================================
// Coarse-to-fine template match producing a per-ROI displacement seed.
// Replaces the single-scale full-resolution search: same capture range,
// far less computation, fewer ambiguous peaks (downsampling smooths away
// fine periodic speckle structure).
//
//   levels=3 -> full, /2, /4. Coarsest level divides motion by 2^(levels-1),
//   so a 500px motion becomes ~125px at /4 — easily found in a small window.
//
// Returns true if the final response clears PHASE_CORR_RESPONSE_ABORT.
// out_u/out_v are in full-resolution pixels; out_resp is the finest-level ZNCC.
static bool pyramidTemplateMatch(
    const cv::Mat& ref_gray, const cv::Mat& def_gray,
    int cx, int cy, int t_half, int levels,
    float& out_u, float& out_v, float& out_resp)
{
    std::vector<cv::Mat> ref_pyr{ ref_gray };
    std::vector<cv::Mat> def_pyr{ def_gray };
    for (int l = 1; l < levels; ++l) {
        cv::Mat rd, dd;
        cv::pyrDown(ref_pyr.back(), rd);
        cv::pyrDown(def_pyr.back(), dd);
        ref_pyr.push_back(rd);
        def_pyr.push_back(dd);
    }

    float est_u = 0.0f, est_v = 0.0f, resp = 0.0f;
    bool any_level_matched = false;

    for (int l = levels - 1; l >= 0; --l) {
        float scale = 1.0f / (float)(1 << l);   // 1 / 2^l
        const cv::Mat& R = ref_pyr[l];
        const cv::Mat& D = def_pyr[l];

        int lcx = (int)std::round(cx * scale);
        int lcy = (int)std::round(cy * scale);
        int lth = std::max(8, (int)std::round(t_half * scale));

        int tx = std::max(0, lcx - lth);
        int ty = std::max(0, lcy - lth);
        int tw = std::min(2 * lth, R.cols - tx);
        int th = std::min(2 * lth, R.rows - ty);
        if (tw < 8 || th < 8) continue;

        cv::Mat tmpl = R(cv::Rect(tx, ty, tw, th)).clone();
        cv::Scalar mn, sd;
        cv::meanStdDev(tmpl, mn, sd);
        if (sd[0] < 6.0) continue;   // featureless at this scale

        // Coarsest level: search the full (downsampled) capture range.
        // Finer levels: only refine around the running estimate (few px).
        int est_lx = (int)std::round(est_u * scale);
        int est_ly = (int)std::round(est_v * scale);
        int margin = (l == levels - 1)
            ? (Config::GLOBAL_SHIFT_SEARCH_MARGIN_PX >> (levels - 1)) + 8
            : 6;

        int sx = std::max(0, tx + est_lx - margin);
        int sy = std::max(0, ty + est_ly - margin);
        int sw = std::min(D.cols - sx, tw + 2 * margin);
        int sh = std::min(D.rows - sy, th + 2 * margin);
        if (sw <= tw || sh <= th) continue;

        cv::Mat search = D(cv::Rect(sx, sy, sw, sh));
        cv::Mat mmap;
        cv::matchTemplate(search, tmpl, mmap, cv::TM_CCOEFF_NORMED);
        if (mmap.empty()) continue;

        double minV, maxV; cv::Point minL, maxL;
        cv::minMaxLoc(mmap, &minV, &maxV, &minL, &maxL);

        float lvl_u = (float)((sx + maxL.x) - tx);
        float lvl_v = (float)((sy + maxL.y) - ty);
        est_u = lvl_u / scale;
        est_v = lvl_v / scale;
        resp = (float)maxV;
        any_level_matched = true;
    }

    out_u = est_u;
    out_v = est_v;
    out_resp = resp;
    return any_level_matched && resp >= (float)Config::PHASE_CORR_RESPONSE_ABORT;
}


// =============================================================================
// buildROIDefinitions
// =============================================================================

std::array<ROIDefinition, 5> buildROIDefinitions(int img_width, int img_height)
{
    std::array<ROIDefinition, 5> defs;

    EffectiveROI roi = resolveEffectiveROI();
    int cx = roi.cx;
    int cy = roi.cy;
    int optic_r = roi.radius_px;
    if (roi.from_detection)
        std::cout << "[ROI-DEF] Using detected optic centre/radius.\n";

    // Placement radius for P/Q/R/S
    int placement_r = (int)(optic_r * Config::ROI_PLACEMENT_FRACTION);

    int hs = Config::ROI_HALF_SIZE_PX;
    int gs = Config::ROI_GRID_STEP;
    int sr = Config::ROI_SUBSET_RADIUS;

    // Order: 0=C, 1=P, 2=Q, 3=R, 4=S
    struct Pos { int x; int y; const char* label; };
    std::array<Pos, 5> positions = { {
        { cx,               cy,               "C" },
        { cx,               cy - placement_r, "P" },
        { cx + placement_r, cy,               "Q" },
        { cx,               cy + placement_r, "R" },
        { cx - placement_r, cy,               "S" }
    } };

    for (int i = 0; i < 5; ++i) {
        defs[i].label = positions[i].label;
        defs[i].centre_x = positions[i].x;
        defs[i].centre_y = positions[i].y;
        defs[i].half_size = hs;
        defs[i].grid_step = gs;
        defs[i].subset_radius = sr;

        int margin = sr + 2;
        int clamped_x = std::clamp(defs[i].centre_x,
            hs + margin, img_width - hs - margin);
        int clamped_y = std::clamp(defs[i].centre_y,
            hs + margin, img_height - hs - margin);

        // If the clamp moved the centre by more than half the patch size,
        // the ROI is no longer sitting on the optic at the intended radius.
        // Tilt and rotation biomarkers depend on the lever arm being correct —
        // flag this so the user knows to adjust ROI_CENTER or ROI_PLACEMENT_FRACTION.
        int dx = std::abs(clamped_x - defs[i].centre_x);
        int dy = std::abs(clamped_y - defs[i].centre_y);
        if (dx > 0 || dy > 0) {
            std::cout << "[ROI-DEF] WARNING: ROI-" << defs[i].label
                << " intended centre (" << defs[i].centre_x
                << ", " << defs[i].centre_y
                << ") is outside image bounds. Clamped to ("
                << clamped_x << ", " << clamped_y << ").\n";
            if (dx > hs / 2 || dy > hs / 2)
                std::cout << "[ROI-DEF] ERROR: ROI-" << defs[i].label
                << " clamped by more than half its patch size ("
                << dx << "px, " << dy << "px). "
                << "Tilt/rotation measurements will be wrong. "
                << "Reduce ROI_PLACEMENT_FRACTION or adjust ROI_CENTER.\n";
            defs[i].clamped = true;
        }

        defs[i].centre_x = clamped_x;
        defs[i].centre_y = clamped_y;
    }
    std::cout << "[ROI-DEF] Optic radius    : " << optic_r << " px\n";
    std::cout << "[ROI-DEF] Placement radius: " << placement_r << " px  ("
        << Config::ROI_PLACEMENT_FRACTION * 100.0f << "% of optic)\n";
    std::cout << "[ROI-DEF] ROI half-size   : " << hs << " px\n";
    for (const auto& d : defs) {
        std::cout << "[ROI-DEF]   " << d.label
            << "  centre=(" << d.centre_x << ", " << d.centre_y << ")"
            << (d.clamped ? "  [CLAMPED]" : "") << "\n";
    }

    return defs;
}


// =============================================================================
// buildROIPOIGrid
// =============================================================================
// Builds a POI grid for one ROI — a square patch centred at (cx, cy).

static std::vector<POI2D> buildROIPOIGrid(const ROIDefinition& def,
    int img_width, int img_height)
{
    std::vector<POI2D> queue;

    int cx = def.centre_x;
    int cy = def.centre_y;
    int hs = def.half_size;
    int step = def.grid_step;
    int margin = def.subset_radius + 2;

    int x_start = std::max(margin, cx - hs);
    int x_end = std::min(img_width - margin, cx + hs);
    int y_start = std::max(margin, cy - hs);
    int y_end = std::min(img_height - margin, cy + hs);

    for (int y = y_start; y <= y_end; y += step)
        for (int x = x_start; x <= x_end; x += step)
            queue.emplace_back(Point2D((float)x, (float)y));

    return queue;
}


// =============================================================================
// createROIDICContext
// =============================================================================

ROIDICContext createROIDICContext(const std::string& ref_path,
    int img_width, int img_height)
{
    ROIDICContext rctx;
    rctx.defs = buildROIDefinitions(img_width, img_height);

    int n_threads = Config::OMP_THREADS;
    if (n_threads <= 0) n_threads = omp_get_max_threads();

    std::cout << "[ROI-CTX] Building 5 DIC contexts ...\n";

    for (int i = 0; i < 5; ++i) {
        const auto& def = rctx.defs[i];

        rctx.ctx[i].ref_path = ref_path;
        rctx.ctx[i].ref_img = new Image2D(ref_path);
        rctx.ctx[i].n_threads = n_threads;

        rctx.ctx[i].fftcc = new FFTCC2D(def.subset_radius,
            def.subset_radius,
            n_threads);

        rctx.ctx[i].icgn = new ICGN2D1(def.subset_radius,
            def.subset_radius,
            Config::ICGN_CONVERGENCE,
            Config::ICGN_MAX_ITER,
            n_threads);

        // Hessian computed over the full reference image — ICGN needs this
        // even though we only correlate a small patch. The Hessian is
        // pre-computed for every pixel; we just use the subset of those
        // that fall inside our ROI patch at runtime.
        rctx.ctx[i].icgn->setImages(*rctx.ctx[i].ref_img, *rctx.ctx[i].ref_img);
        rctx.ctx[i].icgn->prepareRef();

        std::cout << "[ROI-CTX]   " << def.label
            << " ready  (" << def.centre_x << ", " << def.centre_y << ")\n";
    }

    rctx.ready = true;
    std::cout << "[ROI-CTX] All 5 contexts ready.\n";
    return rctx;
}


// =============================================================================
// destroyROIDICContext
// =============================================================================

void destroyROIDICContext(ROIDICContext& rctx)
{
    for (auto& c : rctx.ctx) {
        delete c.fftcc;
        delete c.icgn;
        delete c.ref_img;
        c.fftcc = nullptr;
        c.icgn = nullptr;
        c.ref_img = nullptr;
    }
    rctx.ready = false;
}


// =============================================================================
// runOneROI  (internal)
// =============================================================================
// FFTCC coarse match + first-order ICGN refinement for one ROI patch.
// Every POI is seeded with the ROI's pyramid displacement plus the GLOBAL
// seed-gradient estimate (passed in), which carries rotation into the initial
// guess so ICGN converges in 1-2 iterations even at large lever arms.
// Returns median (u, v) of valid POIs and quality metrics.

static ROIPointResult runOneROI(DICContext& ctx,
    const ROIDefinition& def,
    Image2D& def_img,
    float gs_u, float gs_v,
    float seed_ux, float seed_uy, float seed_vx, float seed_vy)
{
    ROIPointResult r;
    r.label = def.label;


    const bool verbose = Config::PRINT_VERBOSE_DIC;   
    const float roi_cx = (float)def.centre_x;          
    const float roi_cy = (float)def.centre_y;


    auto poi_queue = buildROIPOIGrid(def, ctx.ref_img->width,
        ctx.ref_img->height);

    // Pre-initialise with the ROI seed (zeroth-order; gradients set below)
    for (auto& poi : poi_queue) {
        poi.deformation.u = gs_u;
        poi.deformation.v = gs_v;
    }

    // Boundary filter — remove POIs whose shifted subset leaves the image
    int margin = def.subset_radius + 1;
    poi_queue.erase(
        std::remove_if(poi_queue.begin(), poi_queue.end(),
            [&](const POI2D& poi) {
                float sx = poi.x + gs_u;
                float sy = poi.y + gs_v;
                return sx - margin < 0 ||
                    sx + margin >= (float)ctx.ref_img->width ||
                    sy - margin < 0 ||
                    sy + margin >= (float)ctx.ref_img->height;
            }),
        poi_queue.end());

    if (poi_queue.empty()) {
        r.valid = false;
        r.fail_reason = "All POIs removed by boundary filter — shift too large";
        return r;
    }

    // FFTCC coarse match — zeroth-order per-POI u/v
    ctx.fftcc->setImages(*ctx.ref_img, def_img);
    ctx.fftcc->compute(poi_queue);

    // Coarse divergence filter against the ROI seed — removes gross FFTCC
    // failures only. Kept loose (3x subset radius) so rotation-shifted edge
    // POIs survive; the propagation-aware filter below does the tight gate.
    float coarse_threshold = (float)def.subset_radius * 3.0f;
    poi_queue.erase(
        std::remove_if(poi_queue.begin(), poi_queue.end(),
            [&](const POI2D& poi) {
                float du = poi.deformation.u - gs_u;
                float dv = poi.deformation.v - gs_v;
                return std::sqrt(du * du + dv * dv) > coarse_threshold;
            }),
        poi_queue.end());

    if ((int)poi_queue.size() < 5) {
        r.total_pois = (int)poi_queue.size();
        r.valid = false;
        r.fail_reason = "Too few POIs survived coarse FFTCC filter ("
            + std::to_string(r.total_pois) + ")";
        return r;
    }

    // ── Propagation source: GLOBAL seed-gradient estimate ─────────────────────
    // The rotation/strain gradient comes from the finite difference across the
    // five ROI seeds (computed in runROIDIC and passed in). This is far better
    // conditioned than a single-POI anchor solve, which is unstable at large
    // lever arms. Propagation origin is the ROI centre; displacement there is
    // the ROI's own pyramid seed.
    const float a_x = roi_cx, a_y = roi_cy;
    const float a_u = gs_u, a_v = gs_v;
    const float a_ux = seed_ux, a_uy = seed_uy;
    const float a_vx = seed_vx, a_vy = seed_vy;

    // ── Propagation-aware divergence filter ───────────────────────────────────
    // Predict each POI from the first-order field and reject FFTCC results that
    // deviate from the prediction by more than the tight threshold.
    float tight_threshold = (float)def.subset_radius * 2.0f;
    poi_queue.erase(
        std::remove_if(poi_queue.begin(), poi_queue.end(),
            [&](const POI2D& poi) {
                float dx = poi.x - a_x, dy = poi.y - a_y;
                float u_pred = a_u + a_ux * dx + a_uy * dy;
                float v_pred = a_v + a_vx * dx + a_vy * dy;
                float du = poi.deformation.u - u_pred;
                float dv = poi.deformation.v - v_pred;
                return std::sqrt(du * du + dv * dv) > tight_threshold;
            }),
        poi_queue.end());

    r.total_pois = (int)poi_queue.size();
    if (r.total_pois < 5) {
        r.valid = false;
        r.fail_reason = "Too few POIs survived propagation filter ("
            + std::to_string(r.total_pois) + ")";
        return r;
    }

    // ── Seed every POI with the propagated first-order warp, then ICGN ────────
    for (auto& poi : poi_queue) {
        poi.deformation.ux = a_ux;
        poi.deformation.uy = a_uy;
        poi.deformation.vx = a_vx;
        poi.deformation.vy = a_vy;
        poi.result.zncc = 0.0f;   // clear FFTCC sentinel so ICGN runs the POI
    }

    ctx.icgn->setImages(*ctx.ref_img, def_img);
    ctx.icgn->prepareTar();
    ctx.icgn->compute(poi_queue);

    // ── Collect valid POIs ────────────────────────────────────────────────────
    // Validity: ZNCC above threshold, finite, and within a generous absolute
    // bound of the ROI seed. ICGN has refined each POI independently, so we
    // judge its own output, not the prediction.
    std::vector<double> u_vals, v_vals;
    double zncc_sum = 0.0;
    float plausible_range = (float)def.subset_radius * 4.0f;
    int maxed = 0;
    double iter_sum = 0.0;

    for (const auto& poi : poi_queue) {
        iter_sum += poi.result.iteration;
        if (poi.result.iteration >= Config::ICGN_MAX_ITER) ++maxed;
        bool ok = poi.result.zncc >= Config::ZNCC_MIN_VALID
            && std::isfinite(poi.deformation.u)
            && std::isfinite(poi.deformation.v)
            && std::abs(poi.deformation.u - gs_u) < plausible_range
            && std::abs(poi.deformation.v - gs_v) < plausible_range;
        if (ok) {
            u_vals.push_back(poi.deformation.u);
            v_vals.push_back(poi.deformation.v);
            zncc_sum += poi.result.zncc;
        }
    }

    r.valid_pois = (int)u_vals.size();
    r.valid_fraction = (r.total_pois > 0)
        ? (double)r.valid_pois / r.total_pois : 0.0;
    r.mean_zncc = (r.valid_pois > 0) ? zncc_sum / r.valid_pois : 0.0;

    if (verbose) {
        std::cout << "[ROI-DBG] " << def.label
            << "  pois=" << r.total_pois
            << "  valid=" << r.valid_pois
            << "  meanZNCC=" << std::fixed << std::setprecision(3) << r.mean_zncc
            << "  iters(mean=" << iter_sum / std::max(1, (int)poi_queue.size())
            << ", maxed=" << maxed << ")\n";
    }

    // ── Quality gates ─────────────────────────────────────────────────────────
    if (r.valid_fraction < Config::ROI_MIN_VALID_FRACTION) {
        r.valid = false;
        r.fail_reason = "Valid fraction "
            + std::to_string(r.valid_fraction)
            + " < " + std::to_string(Config::ROI_MIN_VALID_FRACTION);
        return r;
    }
    if (r.mean_zncc < Config::ROI_MIN_MEAN_ZNCC) {
        r.valid = false;
        r.fail_reason = "Mean ZNCC "
            + std::to_string(r.mean_zncc)
            + " < " + std::to_string(Config::ROI_MIN_MEAN_ZNCC);
        return r;
    }

    // Median displacement — robust against residual ICGN outliers
    std::sort(u_vals.begin(), u_vals.end());
    std::sort(v_vals.begin(), v_vals.end());
    size_t mid = u_vals.size() / 2;
    r.u_mean = u_vals[mid];
    r.v_mean = v_vals[mid];
    r.valid = true;
    return r;
}

// =============================================================================
// applyGeometricInterpolation  (internal)
// =============================================================================
// If exactly one cardinal ROI (P/Q/R/S) is invalid, infer its displacement
// from the centre ROI and the opposite cardinal ROI using rigid body constraint.
//
// For a rigid body rotating and translating:
//   u_C ? (u_P + u_R) / 2   and   u_C ? (u_Q + u_S) / 2
// Therefore:
//   u_missing = 2 * u_C - u_opposite

static void applyGeometricInterpolation(std::array<ROIPointResult, 5>& rois,
    int& interpolated_count)
{
    // Index map: 0=C, 1=P, 2=Q, 3=R, 4=S
    // Opposite pairs: P(1)?R(3),  Q(2)?S(4)
    struct Pair { int a; int b; };
    std::array<Pair, 2> pairs = { { {1, 3}, {2, 4} } };

    for (const auto& pair : pairs) {
        int ia = pair.a;
        int ib = pair.b;

        bool a_valid = rois[ia].valid;
        bool b_valid = rois[ib].valid;
        bool c_valid = rois[0].valid;   // centre

        if (a_valid && b_valid) continue;   // both fine
        if (!c_valid)           continue;   // can't interpolate without centre
        if (!a_valid && !b_valid) continue; // can't interpolate if both missing

        if (!a_valid && b_valid) {
            // Infer a from centre and b
            rois[ia].u_mean = 2.0 * rois[0].u_mean - rois[ib].u_mean;
            rois[ia].v_mean = 2.0 * rois[0].v_mean - rois[ib].v_mean;
            rois[ia].valid = true;
            rois[ia].interpolated = true;
            ++interpolated_count;
            std::cout << "[ROI-INTERP] WARNING: ROI-" << rois[ia].label
                << " failed (" << rois[ia].fail_reason << ").\n"
                << "[ROI-INTERP]   Interpolated from C and "
                << rois[ib].label << "  ?  u="
                << std::fixed << std::setprecision(3) << rois[ia].u_mean
                << "  v=" << rois[ia].v_mean << "\n";
        }
        else if (a_valid && !b_valid) {
            // Infer b from centre and a
            rois[ib].u_mean = 2.0 * rois[0].u_mean - rois[ia].u_mean;
            rois[ib].v_mean = 2.0 * rois[0].v_mean - rois[ia].v_mean;
            rois[ib].valid = true;
            rois[ib].interpolated = true;
            ++interpolated_count;
            std::cout << "[ROI-INTERP] WARNING: ROI-" << rois[ib].label
                << " failed (" << rois[ib].fail_reason << ").\n"
                << "[ROI-INTERP]   Interpolated from C and "
                << rois[ia].label << "  ?  u="
                << std::fixed << std::setprecision(3) << rois[ib].u_mean
                << "  v=" << rois[ib].v_mean << "\n";
        }
    }

    // Special case: centre missing, infer from four cardinals if all valid
    if (!rois[0].valid &&
        rois[1].valid && rois[2].valid && rois[3].valid && rois[4].valid)
    {
        rois[0].u_mean = (rois[1].u_mean + rois[2].u_mean +
            rois[3].u_mean + rois[4].u_mean) / 4.0;
        rois[0].v_mean = (rois[1].v_mean + rois[2].v_mean +
            rois[3].v_mean + rois[4].v_mean) / 4.0;
        rois[0].valid = true;
        rois[0].interpolated = true;
        ++interpolated_count;
        std::cout << "[ROI-INTERP] WARNING: ROI-C failed. "
            << "Interpolated from P+Q+R+S mean.\n";
    }
}


// =============================================================================
// runROIDIC
// =============================================================================

ROIDICResult runROIDIC(ROIDICContext& rctx,
    const std::string& ref_path,
    const std::string& def_path,
    int frame_index)
{
    double t0 = nowSec();
    ROIDICResult result;

    std::cout << "\n[ROI-DIC] Frame " << std::setw(4) << std::setfill('0')
        << frame_index << " ????????????????????????????????????\n";

    // ?? Load deformed image once — shared across all 5 ROIs ??????????????????
    Image2D def_img(def_path);

    // ── Global shift pre-initialisation ──────────────────────────────────────
        // Uses the same 5-point multi-ROI median as the full-field pipeline.
        // This is more robust than a single centre-template match — one bad match
        // is outvoted by the other four, and the ambiguity check fires correctly.
    cv::Mat ref_gray = cv::imread(ref_path, cv::IMREAD_GRAYSCALE);
    cv::Mat def_gray = cv::imread(def_path, cv::IMREAD_GRAYSCALE);

    float gs_u = 0.0f, gs_v = 0.0f;
    float gs_response = 0.0f;
    bool  gs_reliable = false;


    float roi_seed_u[5];
    float roi_seed_v[5];
    bool  roi_seed_ok[5] = { false, false, false, false, false };

    if (!ref_gray.empty() && !def_gray.empty()) {
        int t_half = Config::ROI_HALF_SIZE_PX;
        int search_margin = Config::GLOBAL_SHIFT_SEARCH_MARGIN_PX;
        int w = ref_gray.cols, h = ref_gray.rows;

        std::vector<float> valid_u, valid_v, valid_resp;

        for (int i = 0; i < 5; ++i) {
            int cx = rctx.defs[i].centre_x;
            int cy = rctx.defs[i].centre_y;

            // Skip ROIs too close to the edge to host a template
            int probe_half = Config::ROI_HALF_SIZE_PX;
            if (cx - probe_half < 0 || cx + probe_half >= w ||
                cy - probe_half < 0 || cy + probe_half >= h) {
                std::cout << "[ROI-DIC]   Seed ROI-" << rctx.defs[i].label
                    << " skipped (too close to image edge)\n";
                continue;
            }

            float u, v, resp;
            bool matched = pyramidTemplateMatch(
                ref_gray, def_gray, cx, cy, t_half,
                Config::SEED_PYRAMID_LEVELS,
                u, v, resp);

            if (!matched) {
                std::cout << "[ROI-DIC]   Seed ROI-" << rctx.defs[i].label
                    << " skipped (pyramid ZNCC=" << resp << ")\n";
                continue;
            }

            valid_u.push_back(u);
            valid_v.push_back(v);
            valid_resp.push_back(resp);

            roi_seed_u[i] = u;
            roi_seed_v[i] = v;
            roi_seed_ok[i] = true;

            std::cout << "[ROI-DIC]   Seed ROI-" << rctx.defs[i].label
                << "  u=" << u << "  v=" << v
                << "  ZNCC=" << resp << "  OK (pyramid)\n";
        }

        if ((int)valid_u.size() >= 3) {
            std::sort(valid_u.begin(), valid_u.end());
            std::sort(valid_v.begin(), valid_v.end());
            size_t mid = valid_u.size() / 2;
            gs_u = valid_u[mid];
            gs_v = valid_v[mid];
            float resp_sum = 0.f;
            for (float r : valid_resp) resp_sum += r;
            gs_response = resp_sum / (float)valid_resp.size();
            gs_reliable = true;
        }
        else if (!valid_u.empty()) {
            std::sort(valid_u.begin(), valid_u.end());
            std::sort(valid_v.begin(), valid_v.end());
            size_t mid = valid_u.size() / 2;
            gs_u = valid_u[mid];
            gs_v = valid_v[mid];
            // Average resp since valid_resp isn't sorted with u/v
            float resp_sum = 0.f;
            for (float rr : valid_resp) resp_sum += rr;
            gs_response = resp_sum / (float)valid_resp.size();
            gs_reliable = false;
            std::cout << "[ROI-DIC] WARNING: Only " << valid_u.size()
                << "/5 seed templates matched. Shift estimate unreliable.\n";
        }
        else {
            std::cout << "[ROI-DIC] WARNING: No seed templates matched. "
                << "Running FFTCC from zero.\n";
            gs_reliable = false;
        }
    }

    ref_gray.release();
    def_gray.release();

    result.global_shift_u = gs_u;
    result.global_shift_v = gs_v;
    result.global_shift_response = gs_response;

    std::cout << "[ROI-DIC] Global seed    : u=" << gs_u
        << "px  v=" << gs_v
        << "px  ZNCC=" << gs_response
        << (gs_reliable ? "  (reliable)" : "  (weak)") << "\n";

    // ROIs whose own template match failed inherit the global median seed.
    for (int i = 0; i < 5; ++i) {
        if (!roi_seed_ok[i]) {
            roi_seed_u[i] = gs_u;
            roi_seed_v[i] = gs_v;
            std::cout << "[ROI-DIC]   ROI-" << rctx.defs[i].label
                << " using global fallback seed (own match failed)\n";
        }
    };

    // ── Estimate the global first-order displacement gradient from the five
        // ROI seeds. The seeds sample u,v at C/P/Q/R/S positions; finite differences
        // across the opposite pairs give du/dx, du/dy, dv/dx, dv/dy. This gradient
        // is passed to every ROI as the anchor's initial guess so rotating ROIs
        // converge fast instead of cold-starting from zero gradient.
        // Index: 0=C, 1=P(top), 2=Q(right), 3=R(bottom), 4=S(left)
    float seed_ux = 0, seed_uy = 0, seed_vx = 0, seed_vy = 0;
    {
        // Horizontal pair Q(right) - S(left): separation in x
        float dx_qs = (float)(rctx.defs[2].centre_x - rctx.defs[4].centre_x);
        // Vertical pair P(top) - R(bottom): separation in y
        float dy_pr = (float)(rctx.defs[1].centre_y - rctx.defs[3].centre_y);

        if (std::abs(dx_qs) > 1.0f) {
            seed_ux = (roi_seed_u[2] - roi_seed_u[4]) / dx_qs;  // du/dx
            seed_vx = (roi_seed_v[2] - roi_seed_v[4]) / dx_qs;  // dv/dx
        }
        if (std::abs(dy_pr) > 1.0f) {
            seed_uy = (roi_seed_u[1] - roi_seed_u[3]) / dy_pr;  // du/dy
            seed_vy = (roi_seed_v[1] - roi_seed_v[3]) / dy_pr;  // dv/dy
        }
        std::cout << "[ROI-DIC] Seed gradient : ux=" << seed_ux
            << " uy=" << seed_uy << " vx=" << seed_vx
            << " vy=" << seed_vy << "\n";
    }

    // ?? Run five ROIs ?????????????????????????????????????????????????????????
    for (int i = 0; i < 5; ++i) {
        result.rois[i] = runOneROI(rctx.ctx[i], rctx.defs[i],
            def_img, roi_seed_u[i], roi_seed_v[i],
            seed_ux, seed_uy, seed_vx, seed_vy);
        std::cout << "[ROI-DIC]   " << rctx.defs[i].label
            << (result.rois[i].valid ? "  OK" : "  FAIL")
            << "  u=" << std::fixed << std::setprecision(3)
            << result.rois[i].u_mean
            << "  v=" << result.rois[i].v_mean
            << "  ZNCC=" << result.rois[i].mean_zncc
            << "  valid=" << result.rois[i].valid_pois
            << "/" << result.rois[i].total_pois << "\n";
    }
    result.defs = rctx.defs;

    // ?? Geometric interpolation for failed ROIs ???????????????????????????????
    if (Config::ROI_ALLOW_INTERPOLATION) {
        applyGeometricInterpolation(result.rois, result.interpolated_roi_count);
    }

    // ?? Count failures and set frame validity ?????????????????????????????????
    result.failed_roi_count = 0;
    for (const auto& r : result.rois)
        if (!r.valid) ++result.failed_roi_count;

    result.frame_valid = (result.failed_roi_count <= Config::ROI_MAX_FAILED);

    if (!result.frame_valid) {
        std::cout << "[ROI-DIC] Frame INVALID — "
            << result.failed_roi_count << " ROIs failed "
            << "(max allowed=" << Config::ROI_MAX_FAILED << ")\n";
    }

    result.time_total_sec = nowSec() - t0;
    return result;
}


// =============================================================================
// printROIDICResult
// =============================================================================

void printROIDICResult(const ROIDICResult& r, int frame_index)
{
    std::cout << "\n[ROI-RESULT] Frame " << frame_index << "\n";
    std::cout << "[ROI-RESULT] "
        << (r.frame_valid ? "VALID" : "INVALID")
        << "  failed=" << r.failed_roi_count
        << "  interpolated=" << r.interpolated_roi_count
        << "  time=" << std::fixed << std::setprecision(2)
        << r.time_total_sec << "s\n";
    for (const auto& roi : r.rois) {
        std::cout << "[ROI-RESULT]   " << roi.label
            << (roi.interpolated ? " [INTERP]" : "         ")
            << "  u=" << std::setw(8) << std::setprecision(4) << roi.u_mean
            << "  v=" << std::setw(8) << roi.v_mean
            << "  zncc=" << roi.mean_zncc
            << (roi.valid ? "" : "  FAIL: " + roi.fail_reason) << "\n";
    }
}