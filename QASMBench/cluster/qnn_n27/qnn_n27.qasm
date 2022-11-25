OPENQASM 2.0;
include "qelib1.inc";
qreg q[27];
creg c[27];
creg meas[27];
h q[0];
ry(5.049892) q[1];
rz(1.8674082) q[1];
ry(0.31855655) q[2];
rz(4.2346929) q[2];
ry(5.5246824) q[3];
rz(0.74384525) q[3];
ry(5.9121999) q[4];
rz(5.8480143) q[4];
ry(5.1201151) q[5];
rz(3.6509833) q[5];
ry(4.4411262) q[6];
rz(3.8300189) q[6];
ry(4.9598202) q[7];
rz(1.1699212) q[7];
ry(0.40883517) q[8];
rz(3.3245963) q[8];
ry(0.4778567) q[9];
rz(3.1949324) q[9];
ry(5.2658809) q[10];
rz(3.6247592) q[10];
ry(2.2914438) q[11];
rz(1.861489) q[11];
ry(5.8567562) q[12];
rz(6.2170715) q[12];
ry(1.1458205) q[13];
rz(5.8504603) q[13];
ryy(0.066965451) q[1],q[2];
rzz(1.3965091) q[1],q[2];
ryy(0.46710133) q[2],q[3];
rzz(2.9146857) q[2],q[3];
ryy(0.68530735) q[3],q[4];
rzz(0.33325432) q[3],q[4];
ryy(3.7355244) q[4],q[5];
rzz(3.6586891) q[4],q[5];
ryy(2.5386819) q[5],q[6];
rzz(5.3036405) q[5],q[6];
ryy(1.5489014) q[6],q[7];
rzz(2.7281115) q[6],q[7];
ryy(1.9986771) q[7],q[8];
rzz(4.7369811) q[7],q[8];
ryy(1.8504317) q[8],q[9];
rzz(5.3646783) q[8],q[9];
ryy(5.1336697) q[9],q[10];
rzz(4.5502594) q[9],q[10];
ryy(3.0991003) q[10],q[11];
rzz(4.2300483) q[10],q[11];
ryy(0.47362412) q[11],q[12];
rzz(1.3223738) q[11],q[12];
ryy(5.4957607) q[12],q[13];
rzz(5.8881842) q[12],q[13];
ryy(5.0477124) q[1],q[2];
rzz(2.6619004) q[1],q[2];
ryy(0.67834303) q[2],q[3];
rzz(5.7221608) q[2],q[3];
ryy(1.6286719) q[3],q[4];
rzz(1.8909274) q[3],q[4];
ryy(4.419609) q[4],q[5];
rzz(5.2307252) q[4],q[5];
ryy(3.6193309) q[5],q[6];
rzz(2.9645104) q[5],q[6];
ryy(5.9746491) q[6],q[7];
rzz(0.27746545) q[6],q[7];
ryy(0.97546668) q[7],q[8];
rzz(6.0424918) q[7],q[8];
ryy(1.9678664) q[8],q[9];
rzz(4.3986174) q[8],q[9];
ryy(2.3669053) q[9],q[10];
rzz(0.30152347) q[9],q[10];
ryy(2.3437425) q[10],q[11];
rzz(4.6476449) q[10],q[11];
ryy(1.7633514) q[11],q[12];
rzz(5.5531982) q[11],q[12];
ryy(5.7647712) q[12],q[13];
rzz(6.2610269) q[12],q[13];
ry(0.47332945) q[14];
rz(1.1571793) q[14];
ry(-2.2792477) q[15];
rz(-2.1747991) q[15];
ry(0.70518309) q[16];
rz(-2.4494522) q[16];
ry(-3.1184896) q[17];
rz(1.4286481) q[17];
ry(0.30122362) q[18];
rz(-0.30281009) q[18];
ry(-1.7532643) q[19];
rz(-2.711581) q[19];
ry(0.052316037) q[20];
rz(-1.3550111) q[20];
ry(-3.0656114) q[21];
rz(2.5229425) q[21];
ry(0.25293795) q[22];
rz(-0.33541866) q[22];
ry(-0.96193701) q[23];
rz(1.7293998) q[23];
ry(-0.28826259) q[24];
rz(1.6692674) q[24];
ry(-1.0070047) q[25];
rz(1.7873362) q[25];
ry(2.9967601) q[26];
rz(-1.135803) q[26];
cswap q[0],q[1],q[14];
cswap q[0],q[2],q[15];
cswap q[0],q[3],q[16];
cswap q[0],q[4],q[17];
cswap q[0],q[5],q[18];
cswap q[0],q[6],q[19];
cswap q[0],q[7],q[20];
cswap q[0],q[8],q[21];
cswap q[0],q[9],q[22];
cswap q[0],q[10],q[23];
cswap q[0],q[11],q[24];
cswap q[0],q[12],q[25];
cswap q[0],q[13],q[26];
h q[0];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9],q[10],q[11],q[12],q[13],q[14],q[15],q[16],q[17],q[18],q[19],q[20],q[21],q[22],q[23],q[24],q[25],q[26];
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
