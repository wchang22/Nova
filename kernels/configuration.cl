#ifndef CONFIGURATION_H
#define CONFIGURATION_H

const int STACK_SIZE = 64;

const float3 LIGHT_POS = { -4, 2.8, 7 };

const int SHININESS = 32;
const float3 DEFAULT_AMBIENT = 0.1;
const float3 DEFAULT_DIFFUSE = 0.4;
const float3 DEFAULT_SPECULAR = 0.4;

const float RAY_EPSILON = 1e-2; // Prevent self-shadowing

#endif // CONFIGURATION_H
