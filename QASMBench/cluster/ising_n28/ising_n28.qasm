OPENQASM 2.0;
include "qelib1.inc";
qreg q[28];
creg c[28];
creg meas[28];
h q[0];
h q[1];
h q[2];
h q[3];
h q[4];
h q[5];
h q[6];
h q[7];
h q[8];
h q[9];
h q[10];
h q[11];
h q[12];
h q[13];
h q[14];
h q[15];
h q[16];
h q[17];
h q[18];
h q[19];
h q[20];
h q[21];
h q[22];
h q[23];
h q[24];
h q[25];
h q[26];
h q[27];
rz(-1.1049346) q[0];
rz(1.1049346) q[1];
rz(1.1049346) q[1];
cx q[0],q[1];
rz(-1.1049346) q[1];
cx q[0],q[1];
rz(-0.72468668) q[2];
rz(0.72468668) q[3];
rz(0.72468668) q[3];
cx q[2],q[3];
rz(-0.72468668) q[3];
cx q[2],q[3];
rz(-1.4288629) q[4];
rz(1.4288629) q[5];
rz(1.4288629) q[5];
cx q[4],q[5];
rz(-1.4288629) q[5];
cx q[4],q[5];
rz(1.6751132) q[6];
rz(-1.6751132) q[7];
rz(-1.6751132) q[7];
cx q[6],q[7];
rz(1.6751132) q[7];
cx q[6],q[7];
rz(1.0373497) q[8];
rz(-1.0373497) q[9];
rz(-1.0373497) q[9];
cx q[8],q[9];
rz(1.0373497) q[9];
cx q[8],q[9];
rz(1.3044758) q[10];
rz(-1.3044758) q[11];
rz(-1.3044758) q[11];
cx q[10],q[11];
rz(1.3044758) q[11];
cx q[10],q[11];
rz(0.7413099) q[12];
rz(-0.7413099) q[13];
rz(-0.7413099) q[13];
cx q[12],q[13];
rz(0.7413099) q[13];
cx q[12],q[13];
rz(0.8630933) q[14];
rz(-0.8630933) q[15];
rz(-0.8630933) q[15];
cx q[14],q[15];
rz(0.8630933) q[15];
cx q[14],q[15];
rz(-1.1169214) q[16];
rz(1.1169214) q[17];
rz(1.1169214) q[17];
cx q[16],q[17];
rz(-1.1169214) q[17];
cx q[16],q[17];
rz(-0.91111811) q[18];
rz(0.91111811) q[19];
rz(0.91111811) q[19];
cx q[18],q[19];
rz(-0.91111811) q[19];
cx q[18],q[19];
rz(-1.1587232) q[20];
rz(1.1587232) q[21];
rz(1.1587232) q[21];
cx q[20],q[21];
rz(-1.1587232) q[21];
cx q[20],q[21];
rz(0.72339072) q[22];
rz(-0.72339072) q[23];
rz(-0.72339072) q[23];
cx q[22],q[23];
rz(0.72339072) q[23];
cx q[22],q[23];
rz(1.9546081) q[24];
rz(-1.9546081) q[25];
rz(-1.9546081) q[25];
cx q[24],q[25];
rz(1.9546081) q[25];
cx q[24],q[25];
rz(-0.36618039) q[26];
rz(0.36618039) q[27];
rz(0.36618039) q[27];
cx q[26],q[27];
rz(-0.36618039) q[27];
cx q[26],q[27];
rz(-0.84815828) q[1];
rz(0.84815828) q[2];
rz(0.84815828) q[2];
cx q[1],q[2];
rz(-0.84815828) q[2];
cx q[1],q[2];
rz(0.89762133) q[3];
rz(-0.89762133) q[4];
rz(-0.89762133) q[4];
cx q[3],q[4];
rz(0.89762133) q[4];
cx q[3],q[4];
rz(0.8355633) q[5];
rz(-0.8355633) q[6];
rz(-0.8355633) q[6];
cx q[5],q[6];
rz(0.8355633) q[6];
cx q[5],q[6];
rz(-0.78222362) q[7];
rz(0.78222362) q[8];
rz(0.78222362) q[8];
cx q[7],q[8];
rz(-0.78222362) q[8];
cx q[7],q[8];
rz(-1.0057915) q[9];
rz(1.0057915) q[10];
rz(1.0057915) q[10];
cx q[9],q[10];
rz(-1.0057915) q[10];
cx q[9],q[10];
rz(-1.2194914) q[11];
rz(1.2194914) q[12];
rz(1.2194914) q[12];
cx q[11],q[12];
rz(-1.2194914) q[12];
cx q[11],q[12];
rz(1.4719388) q[13];
rz(-1.4719388) q[14];
rz(-1.4719388) q[14];
cx q[13],q[14];
rz(1.4719388) q[14];
cx q[13],q[14];
rz(-0.92246519) q[15];
rz(0.92246519) q[16];
rz(0.92246519) q[16];
cx q[15],q[16];
rz(-0.92246519) q[16];
cx q[15],q[16];
rz(-1.5420291) q[17];
rz(1.5420291) q[18];
rz(1.5420291) q[18];
cx q[17],q[18];
rz(-1.5420291) q[18];
cx q[17],q[18];
rz(0.12770177) q[19];
rz(-0.12770177) q[20];
rz(-0.12770177) q[20];
cx q[19],q[20];
rz(0.12770177) q[20];
cx q[19],q[20];
rz(0.67391245) q[21];
rz(-0.67391245) q[22];
rz(-0.67391245) q[22];
cx q[21],q[22];
rz(0.67391245) q[22];
cx q[21],q[22];
rz(1.0798858) q[23];
rz(-1.0798858) q[24];
rz(-1.0798858) q[24];
cx q[23],q[24];
rz(1.0798858) q[24];
cx q[23],q[24];
rz(1.7712903) q[25];
rz(-1.7712903) q[26];
rz(-1.7712903) q[26];
cx q[25],q[26];
rz(1.7712903) q[26];
cx q[25],q[26];
h q[0];
rz(0) q[0];
h q[0];
rz(0) q[0];
h q[1];
rz(0) q[1];
h q[1];
rz(0) q[1];
h q[2];
rz(0) q[2];
h q[2];
rz(0) q[2];
h q[3];
rz(0) q[3];
h q[3];
rz(0) q[3];
h q[4];
rz(0) q[4];
h q[4];
rz(0) q[4];
h q[5];
rz(0) q[5];
h q[5];
rz(0) q[5];
h q[6];
rz(0) q[6];
h q[6];
rz(0) q[6];
h q[7];
rz(0) q[7];
h q[7];
rz(0) q[7];
h q[8];
rz(0) q[8];
h q[8];
rz(0) q[8];
h q[9];
rz(0) q[9];
h q[9];
rz(0) q[9];
h q[10];
rz(0) q[10];
h q[10];
rz(0) q[10];
h q[11];
rz(0) q[11];
h q[11];
rz(0) q[11];
h q[12];
rz(0) q[12];
h q[12];
rz(0) q[12];
h q[13];
rz(0) q[13];
h q[13];
rz(0) q[13];
h q[14];
rz(0) q[14];
h q[14];
rz(0) q[14];
h q[15];
rz(0) q[15];
h q[15];
rz(0) q[15];
h q[16];
rz(0) q[16];
h q[16];
rz(0) q[16];
h q[17];
rz(0) q[17];
h q[17];
rz(0) q[17];
h q[18];
rz(0) q[18];
h q[18];
rz(0) q[18];
h q[19];
rz(0) q[19];
h q[19];
rz(0) q[19];
h q[20];
rz(0) q[20];
h q[20];
rz(0) q[20];
h q[21];
rz(0) q[21];
h q[21];
rz(0) q[21];
h q[22];
rz(0) q[22];
h q[22];
rz(0) q[22];
h q[23];
rz(0) q[23];
h q[23];
rz(0) q[23];
h q[24];
rz(0) q[24];
h q[24];
rz(0) q[24];
h q[25];
rz(0) q[25];
h q[25];
rz(0) q[25];
h q[26];
rz(0) q[26];
h q[26];
rz(0) q[26];
h q[27];
rz(0) q[27];
h q[27];
rz(0) q[27];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9],q[10],q[11],q[12],q[13],q[14],q[15],q[16],q[17],q[18],q[19],q[20],q[21],q[22],q[23],q[24],q[25],q[26],q[27];
measure q[0] -> meas[0];
measure q[1] -> meas[1];
measure q[2] -> meas[2];
measure q[3] -> meas[3];
measure q[4] -> meas[4];
measure q[5] -> meas[5];
measure q[6] -> meas[6];
measure q[7] -> meas[7];
measure q[8] -> meas[8];
measure q[9] -> meas[9];
measure q[10] -> meas[10];
measure q[11] -> meas[11];
measure q[12] -> meas[12];
measure q[13] -> meas[13];
measure q[14] -> meas[14];
measure q[15] -> meas[15];
measure q[16] -> meas[16];
measure q[17] -> meas[17];
measure q[18] -> meas[18];
measure q[19] -> meas[19];
measure q[20] -> meas[20];
measure q[21] -> meas[21];
measure q[22] -> meas[22];
measure q[23] -> meas[23];
measure q[24] -> meas[24];
measure q[25] -> meas[25];
measure q[26] -> meas[26];
measure q[27] -> meas[27];
