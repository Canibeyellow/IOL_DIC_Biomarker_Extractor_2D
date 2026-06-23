#include "calibration_runtime.h"
#include "config.h"

#include <iostream>
#include <fstream>
#include <filesystem>
#include <nlohmann/json.hpp>

namespace fs = std::filesystem;
using json = nlohmann::json;

RuntimeCalibration g_calibration;

void loadCalibration()
{
    g_calibration.valid = false;

    if (!Config::CALIBRATION_JSON_PATH.empty())
    {
        std::ifstream f(Config::CALIBRATION_JSON_PATH);

        if (f.is_open())
        {
            json calib = json::parse(f);

            if (calib.contains("pixel_size_mm_per_px"))
            {
                g_calibration.valid = true;
                g_calibration.pixel_size_mm_per_px =
                    calib["pixel_size_mm_per_px"].get<double>();
                g_calibration.source = Config::CALIBRATION_JSON_PATH;

                std::cout << "[CALIB] Loaded calibration JSON\n";
                std::cout << "[CALIB] Scale = "
                    << g_calibration.pixel_size_mm_per_px
                    << " mm/px\n";
                return;
            }
        }
    }

    if (Config::CALIBRATED_PIXEL_SIZE_MM_PER_PX > 0.0)
    {
        g_calibration.valid = true;
        g_calibration.pixel_size_mm_per_px =
            Config::CALIBRATED_PIXEL_SIZE_MM_PER_PX;
        g_calibration.source = "Config calibrated value";

        std::cout << "[CALIB] Using config calibrated scale\n";
        return;
    }
    else {
        g_calibration.valid = false;
        g_calibration.pixel_size_mm_per_px = Config::PIXEL_SIZE_MM_PER_PX;
        g_calibration.source = "Estimated lens scale";

        std::cout << "[CALIB] WARNING: using estimated scale only\n";
    }
   
}