#ifndef KERNEL_BRDF_HPP
#define KERNEL_BRDF_HPP

#include "kernels/backend/kernel.hpp"
#include "kernels/backend/math_constants.hpp"
#include "kernels/backend/vector.hpp"
#include "kernels/random.hpp"
#include "kernels/transforms.hpp"

/*
 * https://learnopengl.com/PBR/Theory
 * https://learnopengl.com/PBR/Lighting
 * https://schuttejoe.github.io/post/ggximportancesamplingpart1/
 * Online Computer Graphics II Course: Rendering: Importance Sampling and BRDFs: BRDF Sampling: (CSE
 * 168 and CSE 168x)
 */

namespace nova {

class CookTorranceBRDF {
public:
  DEVICE CookTorranceBRDF(const float3& out_dir,
                          const float3& normal,
                          const float3& diffuse,
                          float metallic,
                          float roughness)
    : out_dir(out_dir),
      normal(normal),
      diffuse(diffuse),
      metallic(metallic),
      roughness(roughness) {}

  /**
   * Online Computer Graphics II: Rendering: Importance Sampling and BRDFs: BRDF Sampling
   * Generate ξ_0, ξ_1, ξ_2
   * Use ξ_0 to decide diffuse (>t) or specular (≤t)
   * Generate Φ in [0, 2π]
   * If diffuse, θ=acos(sqrt(ξ_1))
   * If specular θ_m = atan(α*sqrt(ξ_1/(1-ξ_1)))
   */
  DEVICE float3 sample(uint& rng_state) {
    float xi0 = rand(rng_state);
    float xi1 = rand(rng_state);
    float xi2 = rand(rng_state);

    // Estimate k_s, k_d using geometric normal instead of microfacet normal
    f0 = mix(make_vector<float3>(0.04f), diffuse, metallic);
    o_dot_n = max(dot(out_dir, normal), 0.0f);

    float3 est_F = fresnel_schlick(o_dot_n, f0);
    float3 est_k_s = est_F;
    float3 est_k_d = (1.0f - est_k_s) * (1.0f - metallic);

    // Compute ratio of k_s, k_d
    float3 ratio = clamp(est_k_s / (est_k_s + est_k_d), 0.0f, 1.0f);
    // Clamp at 0.25 to allow for some specular
    float est_t = max((ratio.x + ratio.y + ratio.z) / 3.0f, 0.25f);

    Mat3x3 ray_basis = transpose(create_basis(normal));
    float phi = 2.0f * M_PI_F * xi1;

    // Sample for random microfacet normal, aka half dir

    // Specular
    if (xi0 <= est_t) {
      float theta = atan(roughness * sqrt(xi2 / (1.0f - xi2)));
      m_normal = ray_basis * spherical_to_cartesian(theta, phi);
      in_dir = reflect(-out_dir, m_normal);
    }
    // Diffuse
    else {
      float theta = acos(sqrt(xi2));
      in_dir = ray_basis * spherical_to_cartesian(theta, phi);
      m_normal = normalize(in_dir + out_dir);
    }
    return in_dir;
  }

  /**
   * Cook-Torrance BRDF: https://learnopengl.com/PBR/Theory
   * f_r= = k_d*f_lambert + f_CookTorrance
   * f_lambert = c/π
   * f_CookTorrance = DFG/(4(ω_o⋅n)(ω_i⋅n))
   */
  DEVICE float3 eval() {
    compute_intermediates();

    if (i_dot_m <= 0.0f || i_dot_n <= 0.0f) {
      return make_vector<float3>(0.0f);
    }

    float3 f_lambert = diffuse * M_1_PI_F;
    float3 f_cook_torrance = D * F * G / make_non_zero(4.0f * o_dot_n * i_dot_n);

    return k_d * f_lambert + f_cook_torrance;
  }

  /**
   * Online Computer Graphics II: Rendering: Importance Sampling and BRDFs: BRDF Sampling
   * pdf(ω_i, ω_o) = (1-t)(n⋅ω_i)/π + t*D(n⋅ω_m)/(4(ω_i⋅ω_m))
   * t = k_s/(k_d+k_s)
   */
  DEVICE float3 pdf() {
    compute_intermediates();
    return make_non_zero((1.0f - t) * i_dot_n * M_1_PI_F +
                         t * D * n_dot_m / make_non_zero(4.0f * i_dot_m));
  }

private:
  // NDF_GGXTR(n,m,α) = (α^2)/(π((n⋅m)^2(α^2−1)+1)^2)
  DEVICE inline float distribution_ggx(float n_dot_m, float alpha) {
    float alpha_2 = alpha * alpha;
    float n_dot_m_2 = n_dot_m * n_dot_m;
    float x = n_dot_m_2 * (alpha_2 - 1.0f) + 1.0f;
    return alpha_2 / make_non_zero(M_PI_F * x * x);
  }

  // G_SchlickGGX(n,v,k) = (n⋅v)/((n⋅v)(1−k)+k)
  DEVICE inline float geometry_schlick_ggx(float n_dot_v, float k) {
    return n_dot_v / make_non_zero(n_dot_v * (1.0f - k) + k);
  };

  // G(i,o,n,k) = G_sub(n,i,k)*G_sub(n,o,k)
  DEVICE inline float geometry_smith(float i_dot_n, float o_dot_n, float k) {
    return geometry_schlick_ggx(i_dot_n, k) * geometry_schlick_ggx(o_dot_n, k);
  }

  // F_Schlick(i,m,F0) = F0+(1−F0)(1−(i⋅m))^5
  DEVICE inline float3 fresnel_schlick(float i_dot_m, const float3& f0) {
    return f0 + (1.0f - f0) * pow(1.0f - i_dot_m, 5.0f);
  }

  DEVICE void compute_intermediates() {
    if (intermediates_computed) {
      return;
    }

    i_dot_m = dot(in_dir, m_normal);
    i_dot_n = dot(in_dir, normal);
    n_dot_m = max(dot(normal, m_normal), 0.0f);

    float alpha = roughness * roughness;
    D = distribution_ggx(n_dot_m, alpha);

    F = fresnel_schlick(i_dot_m, f0);

    float k_direct = roughness + 1.0f;
    k_direct = k_direct * k_direct / 8.0f;
    G = geometry_smith(i_dot_n, o_dot_n, k_direct);

    float3 k_s = F;
    k_d = (1.0f - k_s) * (1.0f - metallic);
    t = clamp(k_s / (k_s + k_d), 0.0f, 1.0f);

    intermediates_computed = true;
  }

  float3 in_dir;
  float3 out_dir;
  float3 normal;
  float3 m_normal;
  float3 diffuse;
  float metallic;
  float roughness;

  // Intermediates
  bool intermediates_computed = false;
  float D;
  float3 F;
  float G;
  float3 f0;
  float3 k_d;
  float3 t;
  float i_dot_m;
  float i_dot_n;
  float o_dot_n;
  float n_dot_m;
};

}

#endif // KERNEL_BRDF_HPP