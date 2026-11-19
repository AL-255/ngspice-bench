* GF180MCU Typical Analog PDK Include for opa1000_bundle
.include "../pdk/gf180mcu/gf180mcuD/libs.tech/ngspice/design.ngspice"
.param sw_stat_global=0
.param sw_stat_mismatch=0
.lib "../pdk/gf180mcu/gf180mcuD/libs.tech/ngspice/sm141064.ngspice" typical
.lib "../pdk/gf180mcu/gf180mcuD/libs.tech/ngspice/sm141064.ngspice" res_typical
.lib "../pdk/gf180mcu/gf180mcuD/libs.tech/ngspice/sm141064.ngspice" mimcap_typical
.lib "../pdk/gf180mcu/gf180mcuD/libs.tech/ngspice/sm141064.ngspice" cap_mim
.lib "../pdk/gf180mcu/gf180mcuD/libs.tech/ngspice/sm141064.ngspice" moscap_typical
