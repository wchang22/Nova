#ifndef KERNEL_BRDF_HPP
#define KERNEL_BRDF_HPP

#include "kernels/backend/kernel.hpp"
#include "kernels/backend/math_constants.hpp"
#include "kernels/backend/vector.hpp"
#include "kernels/random.hpp"

namespace nova {

// NDF_GGXTR(n,h,α) = (α^2)/(π((n⋅h)^2(α^2−1)+1)^2)
DEVICE float distribution_ggx(const float3& n, const float3& h, float alpha) {
  float alpha_2 = alpha * alpha;
  float n_dot_h = max(dot(n, h), 0.0f);
  float n_dot_h_2 = n_dot_h * n_dot_h;
  float x = n_dot_h_2 * (alpha_2 - 1.0f) + 1.0f;
  return alpha_2 / (M_PI_F * x * x);
}

// G_SchlickGGX(n,v,k) = (n⋅v)/((n⋅v)(1−k)+k)
DEVICE float geometry_schlick_ggx(const float3& n, const float3& v, float k) {
  float n_dot_v = max(dot(n, v), 0.0f);
  return n_dot_v / (n_dot_v * (1.0f - k) + k);
};

// G(n,v,l,k) = G_sub(n,v,k)*G_sub(n,l,k)
DEVICE float geometry_smith(const float3& n, const float3& v, const float3& l, float k) {
  return geometry_schlick_ggx(n, v, k) * geometry_schlick_ggx(n, l, k);
}

// F_Schlick(h,v,F0) = F0+(1−F0)(1−(h⋅v))^5
DEVICE inline float3 fresnel_schlick(const float3& h, const float3& v, const float3& f0) {
  float h_dot_v = max(dot(h, v), 0.0f);
  float x = 1.0f - h_dot_v;
  float x2 = x * x;
  float x5 = x2 * x2 * x;
  return f0 + (1.0f - f0) * x;
}

/**
 * Cook-Torrance BRDF: https://learnopengl.com/PBR/Theory
 * f_r= = k_d*f_lambert + k_s*f_CookTorrance
 * f_lambert = c/π
 * f_CookTorrance = DFG/(4(ω_o⋅n)(ω_i⋅n))
 */
DEVICE float3 brdf_eval(const float3& in_dir,
                        const float3& out_dir,
                        const float3& normal,
                        const float3& diffuse,
                        float metallic,
                        float roughness) {
  float3 half_dir = normalize(in_dir + out_dir);

  float3 f0 = mix(make_vector<float3>(0.04f), diffuse, metallic);
  float k_direct = roughness + 1.0f;
  k_direct = k_direct * k_direct / 8.0f;

  float D = distribution_ggx(normal, half_dir, roughness * roughness);
  float3 F = fresnel_schlick(half_dir, out_dir, f0);
  float G = geometry_smith(normal, out_dir, in_dir, k_direct);

  float n_dot_o = max(dot(normal, out_dir), 0.0f);
  float n_dot_i = max(dot(normal, in_dir), 0.0f);
  float3 f_cook_torrance = D * F * G / max(4.0f * n_dot_o * n_dot_i, 1e-3f);
  float3 f_lambert = diffuse / M_PI_F;

  float3 k_s = F;
  float3 k_d = (1.0f - k_s) * (1.0f - metallic);
  return k_d * f_lambert + f_cook_torrance;
}

/**
 * pdf(ω_i, ω_o) = (1-t)(n⋅ω_i)/π + t*D(n⋅h)/(4(ω_i⋅h))
 * t = k_s/(k_d+k_s)
 */
DEVICE float3 brdf_pdf(const float3& in_dir,
                       const float3& out_dir,
                       const float3& normal,
                       const float3& diffuse,
                       float metallic,
                       float roughness) {
  float3 half_dir = normalize(in_dir + out_dir);

  float3 f0 = mix(make_vector<float3>(0.04f), diffuse, metallic);

  float D = distribution_ggx(normal, half_dir, roughness * roughness);
  float3 F = fresnel_schlick(half_dir, out_dir, f0);

  float3 k_s = F;
  float3 k_d = (1.0f - k_s) * (1.0f - metallic);

  float3 t = k_s / (k_d + k_s);
  float n_dot_i = max(dot(normal, in_dir), 0.0f);
  float n_dot_h = max(dot(normal, half_dir), 0.0f);
  float i_dot_h = max(dot(in_dir, half_dir), 0.0f);

  return (1.0f - t) * n_dot_i / M_PI_F + t * D * n_dot_h / max(4.0f * i_dot_h, 1e-3f);
}

DEVICE float3 brdf_sample(uint& rng_state,
                          const float3& in_dir,
                          const float3& out_dir,
                          const float3& normal,
                          const float3& diffuse,
                          float metallic,
                          float roughness) {
  float3 half_dir = normalize(in_dir + out_dir);

  float3 f0 = mix(make_vector<float3>(0.04f), diffuse, metallic);
  float3 F = fresnel_schlick(half_dir, out_dir, f0);

  float3 k_s = F;
  float3 k_d = (1.0f - k_s) * (1.0f - metallic);
  float3 t = k_s / (k_d + k_s);

  float psi0 = rand(rng_state);
  float psi1 = rand(rng_state);
  float psi2 = rand(rng_state);

  float phi = 2.0f * M_PI_F * psi2;
  Mat3x3 ray_basis = transpose(create_basis(normal));
  float3 random_dir;

  // Sample diffuse
  if (psi0 > length(t) / sqrt(3.0f)) {
    float theta = acos(sqrt(psi1));
    random_dir =
      ray_basis * make_vector<float3>(sin(theta) * cos(phi), cos(theta), -sin(theta) * sin(phi));
  }
  // Sample specular
  else {
    float phi_h = phi;
    float theta_h = atan2(roughness * sqrt(psi1), sqrt(1.0f - psi1));
    float3 h = ray_basis * make_vector<float3>(sin(theta_h) * cos(phi_h), cos(theta_h),
                                               -sin(theta_h) * sin(phi_h));
    random_dir = normalize(h - in_dir);
  }
  return random_dir;
}

}

#endif // KERNEL_BRDF_HPP