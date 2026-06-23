#include "compression_model.h"
#include "config.h"

double getCompressionMm(const SessionData& session, int frame_index)
{
    // Priority 1: per-frame compression from metadata (Pico motor log)
    if (Config::COMPRESSION_FROM_METADATA
        && frame_index < (int)session.compression_log.size()
        && session.compression_log[frame_index] >= 0.0)
    {
        return session.compression_log[frame_index];
    }

    // Priority 2: constant step assumption
    if (Config::SESSION_MODE == Config::SessionMode::COMPRESSION)
        return frame_index * Config::COMPRESSION_STEP_MM;

    return 0.0;   // TRANSLATION mode — no compression
}