// =============================================================================
// dic_pipeline.cpp
// =============================================================================

#include "dic_pipeline.h"
#include "config.h"

#include <iostream>
#include <filesystem>
#include <chrono>
#include <cmath>
#include <omp.h>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <algorithm>
#include "calibration_runtime.h"
#include <map>
#include "optic_runtime.h"
using namespace opencorr;
namespace fs = std::filesystem;

// Convenience timer
static double nowSec() { return omp_get_wtime(); }

// =============================================================================
// buildROIGrid
// =============================================================================
// Creates a regular grid of POI2D points inside the circular ROI.
// Points outside the circle are excluded — this is the computational
// savings of ROI-based DIC vs full-field DIC.
//
// Margin: we keep points at least (subset_radius + 1) pixels from the
// image boundary so that every subset is fully within the image.


static std::vector<POI2D> buildROIGrid(int img_width, int img_height)
{
    std::vector<POI2D> queue;

    EffectiveROI roi = resolveEffectiveROI();
    int cx = roi.cx;
    int cy = roi.cy;
    int radius = roi.radius_px;
    if (roi.from_detection)
        std::cout << "[ROI-DEF] Using detected optic centre/radius.\n";

    int step    = Config::GRID_STEP;
 
    int margin = 2 * std::max(Config::SUBSET_RADIUS_X, Config::SUBSET_RADIUS_Y) + 2;

    // Bounding box of the circular ROI (clamped to image bounds with margin)
    int x_start = std::max(margin,            cx - radius);
    int x_end   = std::min(img_width - margin, cx + radius);
    int y_start = std::max(margin,             cy - radius);
    int y_end   = std::min(img_height - margin, cy + radius);

    int r2 = radius * radius;

    for (int y = y_start; y <= y_end; y += step) {
        for (int x = x_start; x <= x_end; x += step) {
            // Include only points inside the circle
            int dx = x - cx;
            int dy = y - cy;
            if (dx*dx + dy*dy <= r2) {
                POI2D poi(Point2D(static_cast<float>(x),
                                  static_cast<float>(y)));
                queue.push_back(poi);
            }
        }
    }

    return queue;
}


// =============================================================================
// runDIC
// =============================================================================

DICContext createDICContext(const std::string& ref_path)
{
    std::cout << "[CTX] createDICContext entered\n" << std::flush;
    std::cout << "[CTX] ref_path = " << ref_path << "\n" << std::flush;

    DICContext ctx;  
    std::cout << "[CTX] DICContext constructed\n" << std::flush;

    ctx.ref_path = ref_path;                          
    ctx.ref_img = new Image2D(ref_path);
    std::cout << "[CTX] Image2D loaded, w=" << ctx.ref_img->width << "\n" << std::flush;

    // Determined once, stored in context
    int n_threads = Config::OMP_THREADS;
    if (n_threads <= 0)
        n_threads = omp_get_max_threads();

    omp_set_num_threads(n_threads);     // set once here
    ctx.n_threads = n_threads;          // stored for runDIC to read
    std::cout << "[DIC]  CPU threads  : " << n_threads << "\n";

    ctx.fftcc = new FFTCC2D(Config::SUBSET_RADIUS_X,
        Config::SUBSET_RADIUS_Y,
        n_threads);

    ctx.icgn = new ICGN2D1(Config::SUBSET_RADIUS_X,
        Config::SUBSET_RADIUS_Y,
        Config::ICGN_CONVERGENCE,
        Config::ICGN_MAX_ITER,
        n_threads);

    // THIS is the expensive call — only happens once now
    ctx.icgn->setImages(*ctx.ref_img, *ctx.ref_img);  // ref vs ref for prepare
    ctx.icgn->prepare();

    // Validate ROI fits within image
    int img_h = ctx.ref_img->height;
    int img_w = ctx.ref_img->width;
    int margin = std::max(Config::SUBSET_RADIUS_X, Config::SUBSET_RADIUS_Y) + 1;

    // Validate the ROI fits — use the SAME effective centre/radius that
        // buildROIGrid uses, so the diagnostic matches what actually runs.
// Validate ROI fits within image — use the SAME effective centre/radius
    // that buildROIGrid uses, so the diagnostic matches what actually runs.
    EffectiveROI roi = resolveEffectiveROI();
    int roi_r = roi.radius_px;
    int img_h = ctx.ref_img->height;
    int img_w = ctx.ref_img->width;
    int margin = std::max(Config::SUBSET_RADIUS_X, Config::SUBSET_RADIUS_Y) + 1;

    std::cout << "[CTX]  Effective ROI: centre=(" << roi.cx << ", " << roi.cy
        << ")  radius=" << roi_r << " px  ("
        << (roi.from_detection ? "detected optic" : "config/calibration")
        << ")\n";

    if (roi.cx - roi_r < margin || roi.cx + roi_r > img_w - margin ||
        roi.cy - roi_r < margin || roi.cy + roi_r > img_h - margin)
    {
        std::cout << "[CTX]  WARNING: ROI extends outside image bounds.\n"
            << "[CTX]  ROI covers x=[" << roi.cx - roi_r << ", " << roi.cx + roi_r << "]  "
            << "y=[" << roi.cy - roi_r << ", " << roi.cy + roi_r << "]\n"
            << "[CTX]  Image is " << img_w << "x" << img_h << ".\n"
            << "[CTX]  " << (roi.from_detection
                ? "Detected optic ROI clips the frame — recentre the IOL."
                : "Reduce ROI_RADIUS_MM or adjust ROI_CENTER in config.h.") << "\n";
    }

    std::cout << "[DIC]  Context ready. ICGN Hessian computed once.\n";
    return ctx;
}



// =============================================================================
// pyramidTemplateMatch  (shared — see dic_pipeline.h)
// =============================================================================
bool pyramidTemplateMatch(
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
        float scale = 1.0f / (float)(1 << l);
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
        if (sd[0] < 6.0) continue;

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

    out_u = est_u; out_v = est_v; out_resp = resp;
    return any_level_matched && resp >= (float)Config::PHASE_CORR_RESPONSE_ABORT;
}

// =============================================================================
// estimateGlobalShift
// =============================================================================
// Computes a fast rigid-body displacement estimate between two images using
// phase correlation on the full frame. Used to pre-initialise FFTCC so that
// SUBSET_RADIUS only needs to cover the residual error (~±5px) rather than
// the full displacement magnitude.
//
// Returns {shift_u, shift_v, response}.
// response > 0.3 = reliable. response < 0.1 = unreliable, treat with caution.

struct GlobalShift
{
    float u = 0.0f;
    float v = 0.0f;
    float response = 0.0f;
    bool  reliable = true;
    bool  abort = false;

    float ux = 0.0f, uy = 0.0f, vx = 0.0f, vy = 0.0f;
    float anchor_x = 0.0f, anchor_y = 0.0f;
};

// =============================================================================
// estimateGlobalShiftMultiROI  (now mirrors the ROI pipeline exactly)
// =============================================================================
// Pyramid match at the 5 ISO points (C/P/Q/R/S), median of valid matches.
//   >=3 valid : reliable median
//   1-2 valid : weak median (STILL injected — see runDIC)
//   0   valid : run FFTCC from zero (gs stays 0,0; reliable=false)
// Aborts ONLY on an unreadable image, never on a poor match.
static GlobalShift estimateGlobalShiftMultiROI(
    const std::string& ref_path,
    const std::string& def_path,
    std::vector<DICResult::ROIShift>& roi_shifts_out)
{
    GlobalShift gs;                 // u=v=response=0, reliable=true, abort=false
    roi_shifts_out.clear();

    cv::Mat ref_gray = cv::imread(ref_path, cv::IMREAD_GRAYSCALE);
    cv::Mat def_gray = cv::imread(def_path, cv::IMREAD_GRAYSCALE);
    if (ref_gray.empty() || def_gray.empty()) {
        gs.abort = true;            // genuine hard error: image unreadable
        std::cout << "[SHIFT]  ABORT: Could not load images for multi-ROI shift.\n";
        return gs;
    }

    // Match buildROIGrid/buildROIDefinitions: shift templates AND the
        // propagation anchor (gs.anchor_x/y, from cx/cy) must sit on the
        // effective centre.
    EffectiveROI roi = resolveEffectiveROI();
    int cx = roi.cx;
    int cy = roi.cy;
    int optic_r = roi.radius_px;

    int placement_r = (int)(optic_r * Config::ROI_PLACEMENT_FRACTION);

    // Same 5 ISO points + order as buildROIDefinitions: 0=C 1=P 2=Q 3=R 4=S
    struct Pt { const char* label; int x; int y; };
    const Pt pts[5] = {
        { "C", cx,               cy               },
        { "P", cx,               cy - placement_r },  // top
        { "Q", cx + placement_r, cy               },  // right
        { "R", cx,               cy + placement_r },  // bottom
        { "S", cx - placement_r, cy               },  // left
    };

    const int t_half = Config::ROI_HALF_SIZE_PX;   // parity with ROI seeds
    const int w = ref_gray.cols, h = ref_gray.rows;

    std::vector<float> valid_u, valid_v, valid_resp;

    for (int i = 0; i < 5; ++i) {
        DICResult::ROIShift rs;
        rs.label = pts[i].label;
        rs.valid = false;

        if (pts[i].x - t_half < 0 || pts[i].x + t_half >= w ||
            pts[i].y - t_half < 0 || pts[i].y + t_half >= h) {
            std::cout << "[SHIFT]   ROI-" << rs.label
                << " skipped (too close to image edge)\n";
            roi_shifts_out.push_back(rs);
            continue;
        }

        float u, v, resp;
        bool matched = pyramidTemplateMatch(
            ref_gray, def_gray, pts[i].x, pts[i].y, t_half,
            Config::SEED_PYRAMID_LEVELS, u, v, resp);

        rs.u = u; rs.v = v; rs.response = resp; rs.valid = matched;
        roi_shifts_out.push_back(rs);

        std::cout << "[SHIFT]   ROI-" << rs.label
            << "  u=" << u << "  v=" << v << "  ZNCC=" << resp
            << (matched ? "  OK (pyramid)" : "  SKIP") << "\n";

        if (matched) { valid_u.push_back(u); valid_v.push_back(v); valid_resp.push_back(resp); }
    }

    ref_gray.release();
    def_gray.release();

    if (valid_u.empty()) {
        gs.reliable = false;        // gs.u = gs.v = 0  -> FFTCC from zero
        std::cout << "[SHIFT]  WARNING: No ROI matched — running FFTCC from zero.\n";
        return gs;
    }

    std::sort(valid_u.begin(), valid_u.end());
    std::sort(valid_v.begin(), valid_v.end());
    size_t mid = valid_u.size() / 2;
    gs.u = valid_u[mid];
    gs.v = valid_v[mid];

    float resp_sum = 0.0f;
    for (float r : valid_resp) resp_sum += r;
    gs.response = resp_sum / (float)valid_resp.size();

    gs.reliable = ((int)valid_u.size() >= 3);
    if (!gs.reliable)
        std::cout << "[SHIFT]  WARNING: Only " << valid_u.size()
        << "/5 ROIs matched — shift estimate weak (still injected).\n";

    float u_spread = valid_u.back() - valid_u.front();
    float v_spread = valid_v.back() - valid_v.front();
    float max_spread = 2.0f * std::max(Config::SUBSET_RADIUS_X, Config::SUBSET_RADIUS_Y);
    if (u_spread > max_spread || v_spread > max_spread)
        std::cout << "[SHIFT]  WARNING: ROI shifts disagree (u_spread=" << u_spread
        << "px, v_spread=" << v_spread << "px). Non-rigid optic or bad match.\n";
    // ── Global first-order field from the 5 seeds (mirrors the ROI pipeline) ──
    // Finite differences across the opposite ISO pairs give du/dx, du/dy,
    // dv/dx, dv/dy. Points that did not match inherit the robust median, so a
    // missing pair contributes ZERO gradient (safe fallback) rather than noise.
    {
        float s_u[5], s_v[5];
        for (int i = 0; i < 5; ++i) {
            if (roi_shifts_out[i].valid) { s_u[i] = roi_shifts_out[i].u; s_v[i] = roi_shifts_out[i].v; }
            else { s_u[i] = gs.u;                s_v[i] = gs.v; }
        }
        float dx_qs = (float)(pts[2].x - pts[4].x);   // Q(right) - S(left)
        float dy_pr = (float)(pts[1].y - pts[3].y);   // P(top)   - R(bottom)
        if (std::abs(dx_qs) > 1.0f) { gs.ux = (s_u[2] - s_u[4]) / dx_qs; gs.vx = (s_v[2] - s_v[4]) / dx_qs; }
        if (std::abs(dy_pr) > 1.0f) { gs.uy = (s_u[1] - s_u[3]) / dy_pr; gs.vy = (s_v[1] - s_v[3]) / dy_pr; }
        gs.anchor_x = (float)cx;
        gs.anchor_y = (float)cy;
        std::cout << "[SHIFT]  Seed gradient   : ux=" << gs.ux << " uy=" << gs.uy
            << " vx=" << gs.vx << " vy=" << gs.vy << "\n";
    }
    std::cout << "[SHIFT]  Combined shift  : u=" << gs.u << "px  v=" << gs.v
        << "px  (from " << valid_u.size() << "/5 ROIs)"
        << (gs.reliable ? "  (reliable)" : "  (weak)") << "\n";
    return gs;
}

// --------- print health check ------------------------------------------------------
// Check for icgn, global shift, and ftcc failure points
static void printDICHealthCheck(const DICResult& result,
    int              max_iter)
{
    const auto& q = result.poi_queue;
    if (q.empty()) return;

    int    n = (int)q.size();
    int    zero_disp = 0;
    int    iter_maxed = 0;
    int    zncc_sentinel = 0;
    int    zncc_good = 0;
    int    zncc_poor = 0;
    double iter_sum = 0, conv_sum = 0, zncc_sum = 0;
    float  u_min = 1e9, u_max = -1e9, v_min = 1e9, v_max = -1e9;
    int   range_count = 0;

    int icgn_ran = 0;  // ADD this counter

    for (const auto& p : q) {
        if (std::abs(p.deformation.u) < 2500.0f && std::abs(p.deformation.v) < 2500.0f) {
            u_min = std::min(u_min, p.deformation.u);
            u_max = std::max(u_max, p.deformation.u);
            v_min = std::min(v_min, p.deformation.v);
            v_max = std::max(v_max, p.deformation.v);
            ++range_count;
            if (p.deformation.u == 0.0f && p.deformation.v == 0.0f)
                ++zero_disp;
        }

        if (p.result.zncc <= -3.0f) {
            ++zncc_sentinel;
            zncc_sum += p.result.zncc;
            // SKIP iteration/convergence — ICGN never ran for this POI
        }
        else {
            // ICGN ran — include in convergence averages
            iter_sum += p.result.iteration;
            conv_sum += p.result.convergence;
            if (p.result.iteration >= max_iter) ++iter_maxed;
            if (p.result.zncc >= 0.5f)       ++zncc_good;
            else if (p.result.zncc < 0.2f)   ++zncc_poor;
            zncc_sum += p.result.zncc;
            ++icgn_ran;
        }
    }

    std::cout << "[HEALTH] Global shift    : u=" << result.global_shift_u
        << "px  v=" << result.global_shift_v << "px  "
        << "response=" << result.global_shift_response;
    if (result.global_shift_response < 0.1f)
        std::cout << "  ← WEAK — pre-initialisation may be wrong";
    std::cout << "\n";

    std::cout << "\n[HEALTH] ── DIC Frame Diagnostics ──────────────────\n";
    std::cout << "[HEALTH] POIs total          : " << n << "\n";
    std::cout << "[HEALTH] FFTCC zero-disp     : " << zero_disp
        << (zero_disp == n ? "  ← ALL ZERO — image may be identical or flat" : "") << "\n";
    std::cout << "[HEALTH] FFTCC u range       : [" << u_min << ", " << u_max << "] px\n";
    std::cout << "[HEALTH] FFTCC v range       : [" << v_min << ", " << v_max << "] px\n";
    std::cout << "[HEALTH] ICGN mean iters     : "
        << (icgn_ran > 0 ? iter_sum / icgn_ran : 0.0) << "\n";
    std::cout << "[HEALTH] ICGN hit max iter   : " << iter_maxed
        << " (" << 100.0 * iter_maxed / n << "%)"
        << (iter_maxed > n * 0.5 ? "  ← MAJORITY FAILED TO CONVERGE" : "") << "\n";
    std::cout << "[HEALTH] ICGN mean conv delta: "
        << (icgn_ran > 0 ? conv_sum / icgn_ran : 0.0) << "\n";
    std::cout << "[HEALTH] ZNCC mean           : " << zncc_sum / n << "\n";
    std::cout << "[HEALTH] ZNCC sentinel (-4)  : " << zncc_sentinel
        << (zncc_sentinel > n * 0.1 ? "  ← HIGH — check FFTCC output" : "") << "\n";
    std::cout << "[HEALTH] ZNCC good (>0.5)    : " << zncc_good
        << " (" << 100.0 * zncc_good / n << "%)\n";
    std::cout << "[HEALTH] ZNCC poor (<0.2)    : " << zncc_poor
        << " (" << 100.0 * zncc_poor / n << "%)\n";
    std::cout << "[HEALTH] ────────────────────────────────────────────\n\n";
}




DICResult runDIC(DICContext& ctx, const std::string& def_path, int frame_index)
{
    std::cout << "\n[DIC]  Frame " << std::setw(4) << std::setfill('0')
              << frame_index << " ────────────────────────────────────\n";
    std::cout << "[DIC]  Def : " << def_path << "\n";

    DICResult result;

    // ── Load images ───────────────────────────────────────────────────────────
    // Image2D constructor reads the file and converts to float grayscale
    // (CV_32F) internally. This is the only constructor OpenCorr provides.
 
    Image2D def_img(def_path);

    result.img_width  = ctx.ref_img->width;
    result.img_height = ctx.ref_img->height;

    if (def_img.width != ctx.ref_img->width || def_img.height != ctx.ref_img->height) {
        throw std::runtime_error(
            "[DIC] Reference and deformed images have different sizes. "
            "All images in a session must be captured at identical resolution."
        );
    }

    // ── Build POI grid inside circular ROI ────────────────────────────────────
    result.poi_queue = buildROIGrid(ctx.ref_img->width, ctx.ref_img->height);
    std::cout << "[DIC]  POIs in ROI  : " << result.poi_queue.size() << "\n";
    if (result.poi_queue.empty()) {
        throw std::runtime_error(
            "[DIC] ROI produced zero POIs. Check ROI_CENTER_X, ROI_CENTER_Y, "
            "ROI_RADIUS, GRID_STEP, and image size."
        );
    }

// ── Global shift pre-initialisation ──────────────────────────────────────
// Template matching finds the coarse rigid-body displacement.
// This is injected into every POI before FFTCC so FFTCC only needs
// to find the small residual within ±SUBSET_RADIUS, not the full motion.
    GlobalShift gs = estimateGlobalShiftMultiROI(ctx.ref_path, def_path,
        result.roi_shifts);
    result.global_shift_u = gs.u;
    result.global_shift_v = gs.v;
    result.global_shift_response = gs.response;

    // ── Abort path ────────────────────────────────────────────────────────────
    // Template matching failed completely — image is corrupt, occluded,
    // or the IOL moved further than the search margin allows.
    // Flag all POIs as failed and return early — FFTCC and ICGN do not run.
    if (gs.abort) {
        for (auto& poi : result.poi_queue) {
            poi.result.zncc = -5.0f;
            poi.deformation.u = 0.0f;
            poi.deformation.v = 0.0f;
        }
        result.time_fftcc_sec = 0.0;
        result.time_icgn_sec = 0.0;
        result.valid_poi_count = 0;
        std::cout << "[DIC]  Frame aborted — no FFTCC or ICGN will run.\n";
        printDICHealthCheck(result, Config::ICGN_MAX_ITER);
        return result;
    }

    // ── Inject global shift + boundary filter ─────────────────────────────────
    // Pre-initialise every POI with the global shift estimate so FFTCC
    // only needs to find the small residual within ±SUBSET_RADIUS.
    //
    // After injecting, remove any POI whose shifted subset would extend
    // outside the image. Without this filter, FFTCC reads garbage data
    // from out-of-bounds memory and produces million-pixel displacements.
    if (gs.reliable || gs.u != 0.0f || gs.v != 0.0f) {
        for (auto& poi : result.poi_queue) {
            poi.deformation.u = gs.u;
            poi.deformation.v = gs.v;
        }

        int margin = std::max(Config::SUBSET_RADIUS_X, Config::SUBSET_RADIUS_Y) + 1;
        size_t before = result.poi_queue.size();

        result.poi_queue.erase(
            std::remove_if(result.poi_queue.begin(), result.poi_queue.end(),
                [&](const POI2D& poi) {
                    float sx = poi.x + gs.u;
                    float sy = poi.y + gs.v;
                    return sx - margin < 0 ||
                        sx + margin >= (float)ctx.ref_img->width ||
                        sy - margin < 0 ||
                        sy + margin >= (float)ctx.ref_img->height;
                }),
            result.poi_queue.end()
        );

        size_t removed = before - result.poi_queue.size();
        if (removed > 0) {
            std::cout << "[DIC]  Boundary filter removed " << removed
                << " POIs whose shifted subsets exceeded image bounds.\n";
        }
        std::cout << "[DIC]  POIs after boundary check : "
            << result.poi_queue.size() << "\n";

        if (result.poi_queue.empty()) {
            // Every POI was out of bounds — motion is too large for the image
            std::cout << "[DIC]  ERROR: All POIs removed by boundary filter. "
                << "Global shift (" << gs.u << ", " << gs.v
                << ") moves every subset outside the image.\n"
                << "[DIC]  Reduce motion per frame or increase image margins.\n";
            result.time_fftcc_sec = 0.0;
            result.time_icgn_sec = 0.0;
            result.valid_poi_count = 0;
            return result;
        }

        // FFTCC with pre-initialised guess — searches ±SUBSET_RADIUS around gs
        std::cout << "[DIC]  Applied global shift (" << gs.u << ", " << gs.v
            << "px). Running local FFTCC refinement.\n";
        double t0_fftcc = nowSec();
        ctx.fftcc->setImages(*ctx.ref_img, def_img);
        ctx.fftcc->compute(result.poi_queue);
        result.time_fftcc_sec = nowSec() - t0_fftcc;
        std::cout << "[DIC]  FFTCC done   : " << std::fixed
            << std::setprecision(2) << result.time_fftcc_sec << "s\n";
    }

    else {
        std::cout << "[DIC]  WARNING: Global shift weak (ZNCC=" << gs.response
            << "). Running FFTCC from zero — trackable range = ±"
            << Config::SUBSET_RADIUS_X << "px = ±"
            << Config::SUBSET_RADIUS_X * g_calibration.pixel_size_mm_per_px
            << "mm.\n";

        // Still apply boundary filter with zero-shift assumption — POIs near
        // the image edge can read out-of-bounds memory in FFTCC without this.
        int margin = std::max(Config::SUBSET_RADIUS_X, Config::SUBSET_RADIUS_Y) + 1;
        size_t before = result.poi_queue.size();
        result.poi_queue.erase(
            std::remove_if(result.poi_queue.begin(), result.poi_queue.end(),
                [&](const POI2D& poi) {
                    return poi.x - margin < 0 ||
                        poi.x + margin >= (float)ctx.ref_img->width ||
                        poi.y - margin < 0 ||
                        poi.y + margin >= (float)ctx.ref_img->height;
                }),
            result.poi_queue.end()
        );
        size_t removed = before - result.poi_queue.size();
        if (removed > 0)
            std::cout << "[DIC]  Boundary filter (fallback) removed "
            << removed << " edge POIs.\n";

        double t0_fftcc = nowSec();
        ctx.fftcc->setImages(*ctx.ref_img, def_img);
        ctx.fftcc->compute(result.poi_queue);
        result.time_fftcc_sec = nowSec() - t0_fftcc;
        std::cout << "[DIC]  FFTCC done   : " << std::fixed
            << std::setprecision(2) << result.time_fftcc_sec << "s\n";
    }
   

    // ── Filter FFTCC divergence before ICGN ──────────────────────────────────
// POIs whose displacement deviates more than 1.5×SUBSET_RADIUS from the
// global shift estimate are FFTCC failures — noise peaks, not true motion.
// Passing them to ICGN wastes time and produces -nan convergence delta.
    {
        float tight_threshold = std::max(Config::SUBSET_RADIUS_X,
            Config::SUBSET_RADIUS_Y) * 1.5f;
        // Centre the gate on the predicted first-order field, not a constant
                // translation. Under rotation, edge POIs have large but PREDICTABLE
                // displacement; a constant gate would wrongly cull them. This is the
                // ROI pipeline's propagation-aware filter applied to the full grid.
        const float a_x = gs.anchor_x, a_y = gs.anchor_y;
        const float a_u = gs.u, a_v = gs.v;
        const float a_ux = gs.ux, a_uy = gs.uy, a_vx = gs.vx, a_vy = gs.vy;

        size_t before_filter = result.poi_queue.size();
        result.poi_queue.erase(
            std::remove_if(result.poi_queue.begin(), result.poi_queue.end(),
                [&](const POI2D& poi) {
                    float dx = poi.x - a_x, dy = poi.y - a_y;
                    float u_pred = a_u + a_ux * dx + a_uy * dy;
                    float v_pred = a_v + a_vx * dx + a_vy * dy;
                    float du = poi.deformation.u - u_pred;
                    float dv = poi.deformation.v - v_pred;
                    return std::sqrt(du * du + dv * dv) > tight_threshold;
                }),
            result.poi_queue.end()
        );

        size_t fftcc_removed = before_filter - result.poi_queue.size();
        if (fftcc_removed > 0)
            std::cout << "[DIC]  FFTCC divergence filter removed " << fftcc_removed
            << " POIs (threshold=±" << tight_threshold << "px from global shift).\n";

        if (result.poi_queue.size() < (size_t)Config::MIN_POIS_FOR_PLANE) {
            std::cout << "[DIC]  WARNING: Only " << result.poi_queue.size()
                << " POIs survived FFTCC filter — below MIN_POIS_FOR_PLANE.\n"
                << "[DIC]  Skipping ICGN — frame will be marked low-confidence.\n";
            result.valid_poi_count = 0;
            printDICHealthCheck(result, Config::ICGN_MAX_ITER);
            return result;
        }
    }

    // ── ICGN2D1 — sub-pixel refinement ───────────────────────────────────────
    // ICGN2D1 uses a 1st-order shape function (6 parameters: u, v, ux, uy, vx, vy).
    // This models translation + rotation + normal/shear strain within each subset.
    // For IOL rigid-body motion (small rotation, tilt, displacement) this is
    // sufficient — use ICGN2D2 only if you measure haptic strain.
    //
    // prepare() pre-computes the Hessian matrix for every reference subset.
    // This is computed ONCE and reused across all iterations — it is what makes
    // ICGN fast. NEVER skip prepare() — it will produce incorrect results.
    double t0 = nowSec();

        for (auto& poi : result.poi_queue) {
        poi.deformation.ux = gs.ux;
        poi.deformation.uy = gs.uy;
        poi.deformation.vx = gs.vx;
        poi.deformation.vy = gs.vy;
        
    }
    ctx.icgn->setImages(*ctx.ref_img, def_img);
    ctx.icgn->prepareTar();
    ctx.icgn->compute(result.poi_queue);

    // ── Spatial outlier rejection ─────────────────────────────────────────────
// POIs whose displacement deviates strongly from their spatial neighbourhood
// have converged to wrong local minima. Replace with neighbourhood median.
// Uses a 3×3 neighbourhood in grid coordinates.
// Only applied when enough POIs exist to define a meaningful neighbourhood.
    if (result.poi_queue.size() > 50) {
        int step = Config::GRID_STEP;

        // Build a map from (x,y) rounded to grid → POI index for neighbour lookup
        std::map<std::pair<int, int>, int> grid_map;
        for (int i = 0; i < (int)result.poi_queue.size(); ++i) {
            int gx = (int)std::round(result.poi_queue[i].x / step);
            int gy = (int)std::round(result.poi_queue[i].y / step);
            grid_map[{gx, gy}] = i;
        }

        int outliers_replaced = 0;
        for (int i = 0; i < (int)result.poi_queue.size(); ++i) {
            auto& poi = result.poi_queue[i];
            if (poi.result.zncc < Config::ZNCC_MIN_VALID) continue;

            int gx = (int)std::round(poi.x / step);
            int gy = (int)std::round(poi.y / step);

            // Collect valid neighbours in 3×3 grid neighbourhood
            std::vector<float> neighbour_u, neighbour_v;
            for (int dy = -1; dy <= 1; ++dy) {
                for (int dx = -1; dx <= 1; ++dx) {
                    if (dx == 0 && dy == 0) continue;
                    auto it = grid_map.find({ gx + dx, gy + dy });
                    if (it == grid_map.end()) continue;
                    const auto& nb = result.poi_queue[it->second];
                    if (nb.result.zncc >= Config::ZNCC_MIN_VALID) {
                        neighbour_u.push_back(nb.deformation.u);
                        neighbour_v.push_back(nb.deformation.v);
                    }
                }
            }

            if (neighbour_u.size() < 3) continue;

            // Median of neighbours
            std::sort(neighbour_u.begin(), neighbour_u.end());
            std::sort(neighbour_v.begin(), neighbour_v.end());
            float med_u = neighbour_u[neighbour_u.size() / 2];
            float med_v = neighbour_v[neighbour_v.size() / 2];

            // If this POI deviates more than 2×SUBSET_RADIUS from neighbourhood median
            float dev = std::sqrt((poi.deformation.u - med_u) * (poi.deformation.u - med_u) +
                (poi.deformation.v - med_v) * (poi.deformation.v - med_v));
            if (dev > 2.0f * std::max(Config::SUBSET_RADIUS_X, Config::SUBSET_RADIUS_Y)) {
                poi.result.zncc = -3.5f;  // flag as spatial outlier (below ZNCC_MIN_VALID)
                ++outliers_replaced;
            }
        }
        if (outliers_replaced > 0)
            std::cout << "[DIC]  Spatial outlier filter flagged "
            << outliers_replaced << " POIs.\n";
    }

    if (Config::PRINT_VERBOSE_DIC) {
        int zncc_zero = 0, zncc_good_dbg = 0, zncc_low = 0;
        for (const auto& p : result.poi_queue) {
            if (p.result.zncc == 0.0f)      ++zncc_zero;
            else if (p.result.zncc >= 0.4f) ++zncc_good_dbg;
            else                             ++zncc_low;
        }
        std::cout << "[DEBUG] ZNCC: zero=" << zncc_zero
            << " good=" << zncc_good_dbg
            << " low=" << zncc_low << "\n";

        for (int i = 0; i < std::min(10, (int)result.poi_queue.size()); ++i) {
            const auto& p = result.poi_queue[i];
            std::cout << "[POI " << i << "] "
                << "u=" << p.deformation.u
                << " v=" << p.deformation.v
                << " zncc=" << p.result.zncc << "\n";
        }
    }

    // ── Count valid POIs ──────────────────────────────────────────────────────
    // POIs with ZNCC below threshold failed to correlate.
    // This happens on regions with poor texture (outside the speckle coverage).
    // They are kept in the queue but excluded from biomarker computation.
    result.valid_poi_count = 0;
    for (const auto& poi : result.poi_queue) {
        if (isValidPOI(poi))
            ++result.valid_poi_count;
    }

    double valid_pct = 100.0 * result.valid_poi_count / result.poi_queue.size();
    std::cout << "[DIC]  Valid POIs   : " << result.valid_poi_count
              << " / " << result.poi_queue.size()
              << " (" << std::setprecision(1) << valid_pct << "%)\n";

    if (valid_pct < 50.0) {
        std::cout << "[DIC]  WARNING: less than 50% valid POIs. "
                     "Check speckle quality and ROI position.\n";
    }
    printDICHealthCheck(result, Config::ICGN_MAX_ITER);

    return result;
}



// =============================================================================
// saveDICResult
// =============================================================================

void saveDICResult(DICResult&   result,
                   const std::string& def_path,
                   const std::string& output_folder,
                   int                frame_index)
{
    // Create subdirectories
    std::string tables_dir = output_folder + "/tables";
    std::string maps_dir   = output_folder + "/maps";
    fs::create_directories(tables_dir);
    fs::create_directories(maps_dir);

    // Build output paths
    std::ostringstream ss;
    ss << std::setw(4) << std::setfill('0') << frame_index;
    std::string idx_str = ss.str();

    std::string table_path = tables_dir + "/result_" + idx_str + ".csv";
    std::string u_map_path = maps_dir   + "/u_"      + idx_str + ".csv";
    std::string v_map_path = maps_dir   + "/v_"      + idx_str + ".csv";

    // Use OpenCorr's IO2D for standardised CSV output.
    // saveTable2D writes: x, y, u, v, ux, uy, vx, vy, zncc, convergence
    // saveMap2D writes a 2D matrix of the chosen variable (importable as heatmap).
    IO2D io;
    io.setDelimiter(",");
    io.setHeight(result.img_height);
    io.setWidth(result.img_width);

    io.setPath(table_path);
    io.saveTable2D(result.poi_queue);

    OutputVariable out_var = u;
    io.setPath(u_map_path);
    io.saveMap2D(result.poi_queue, out_var);

    out_var = v;
    io.setPath(v_map_path);
    io.saveMap2D(result.poi_queue, out_var);

    std::cout << "[DIC]  Saved table  : " << table_path << "\n";
    std::cout << "[DIC]  Saved u-map  : " << u_map_path << "\n";
    std::cout << "[DIC]  Saved v-map  : " << v_map_path << "\n";
}
void destroyDICContext(DICContext& ctx)
{
    delete ctx.fftcc;
    delete ctx.icgn;
    delete ctx.ref_img;

    ctx.fftcc = nullptr;
    ctx.icgn = nullptr;
    ctx.ref_img = nullptr;
}
