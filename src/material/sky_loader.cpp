#include <algorithm>
#include <cmath>
#include <numeric>
#include <stb_image.h>

#include "sky_loader.hpp"
#include "util/exception/exception.hpp"
#include "util/profiling/profiling.hpp"

namespace nova {

SkyLoader::SkyLoader() : sky({}) { stbi_set_flip_vertically_on_load(true); }

void SkyLoader::load_sky(const std::string& path, bool srgb) {
  image_utils::image<float4> im;

  try {
    im = image_utils::read_image<float4>(path.c_str());
  } catch (const ImageException& e) {
    throw SkyException(e.what());
  }

  if (srgb) {
    constexpr auto gamma_correct = [](float x) -> float {
      return std::pow(x, 2.2f);
    };
    std::for_each(im.data.begin(), im.data.end(), [&](float4& pixel) {
      pixel = { gamma_correct(x(pixel)), gamma_correct(y(pixel)), gamma_correct(z(pixel)),
                w(pixel) };
    });
  }
  sky = std::move(im);
}

image_utils::image<float4> SkyLoader::build() const { return sky; }

}