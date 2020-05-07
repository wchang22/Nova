#ifndef SKY_LOADER_HPP
#define SKY_LOADER_HPP

#include "backend/types.hpp"
#include "util/image/imageutils.hpp"

namespace nova {

class SkyLoader {
public:
  SkyLoader();

  void load_sky(const std::string& path, bool srgb = true);
  image_utils::image<float4> build() const;

private:
  image_utils::image<float4> sky;
};

}

#endif // SKY_LOADER_HPP
