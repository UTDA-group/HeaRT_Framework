.subckt pass_device_output_feedback gnd vdd net2 vo fb
M0 vo net2 vdd vdd pmos l=l_power_p w=w_power_p*1 m=1 nf=1
R1 vo fb res r1
R2 fb gnd res r2
CL vo GND cap cload
.ends pass_device_output_feedback