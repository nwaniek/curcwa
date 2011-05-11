#ifndef __SETUP_HPP__605EC202_1381_44E5_82CE_6294CD63FA10
#define __SETUP_HPP__605EC202_1381_44E5_82CE_6294CD63FA10

#include "types.hpp"

unsigned get_material_count();
void get_materials(const material **m);
void get_wavelengths(float *from, float *to, float *step);
const struct stack* get_stack();
rcwa_approach_t get_method();
order_t get_orders();
float get_incident_angle();
pol_t get_polarization();

#endif /* __SETUP_HPP__605EC202_1381_44E5_82CE_6294CD63FA10 */

