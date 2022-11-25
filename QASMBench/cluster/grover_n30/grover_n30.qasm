// Generated from Cirq v0.8.0

OPENQASM 2.0;
include "qelib1.inc";


// Qubits: [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (5, 0), (6, 0), (7, 0), (8, 0), (9, 0), (10, 0), (11, 0), (12, 0), (13, 0), (14, 0), (15, 0), (16, 0), (17, 0), (18, 0), (19, 0), (20, 0), (21, 0), (22, 0), (23, 0), (24, 0), (25, 0), (26, 0), (27, 0), (28, 0), (29, 0), (30, 0)]
qreg q[31];
creg m_result[30];


x q[30];
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
h q[28];
h q[29];
h q[30];
x q[4];
x q[7];
x q[15];
x q[16];
x q[18];
x q[19];
x q[20];
x q[23];
x q[27];
x q[28];
h q[2];
h q[3];
h q[5];
h q[6];
h q[8];
h q[9];
h q[10];
h q[11];
h q[12];
h q[13];
h q[14];
h q[17];
h q[21];
h q[22];
h q[24];
h q[25];
h q[26];
h q[29];
ccx q[0],q[1],q[30];
x q[4];
x q[7];
x q[15];
x q[16];
x q[18];
x q[19];
x q[20];
x q[23];
x q[27];
x q[28];
x q[2];
x q[3];
x q[5];
x q[6];
x q[8];
x q[9];
x q[10];
x q[11];
x q[12];
x q[13];
x q[14];
x q[17];
x q[21];
x q[22];
x q[24];
x q[25];
x q[26];
x q[29];
h q[0];
h q[1];
h q[4];
h q[7];
h q[15];
h q[16];
h q[18];
h q[19];
h q[20];
h q[23];
h q[27];
h q[28];
x q[2];
x q[3];
x q[5];
x q[6];
x q[8];
x q[9];
x q[10];
x q[11];
x q[12];
x q[13];
x q[14];
x q[17];
x q[21];
x q[22];
x q[24];
x q[25];
x q[26];
x q[29];
x q[0];
x q[1];
x q[4];
x q[7];
x q[15];
x q[16];
x q[18];
x q[19];
x q[20];
x q[23];
x q[27];
x q[28];
h q[2];
h q[3];
h q[5];
h q[6];
h q[8];
h q[9];
h q[10];
h q[11];
h q[12];
h q[13];
h q[14];
h q[17];
h q[21];
h q[22];
h q[24];
h q[25];
h q[26];
h q[29];
h q[1];
x q[4];
x q[7];
x q[15];
x q[16];
x q[18];
x q[19];
x q[20];
x q[23];
x q[27];
x q[28];
cx q[0],q[1];
h q[4];
h q[7];
h q[15];
h q[16];
h q[18];
h q[19];
h q[20];
h q[23];
h q[27];
h q[28];
h q[1];
x q[0];
x q[1];
h q[0];
h q[1];
measure q[0] -> m_result[0];
measure q[1] -> m_result[1];
measure q[2] -> m_result[2];
measure q[3] -> m_result[3];
measure q[4] -> m_result[4];
measure q[5] -> m_result[5];
measure q[6] -> m_result[6];
measure q[7] -> m_result[7];
measure q[8] -> m_result[8];
measure q[9] -> m_result[9];
measure q[10] -> m_result[10];
measure q[11] -> m_result[11];
measure q[12] -> m_result[12];
measure q[13] -> m_result[13];
measure q[14] -> m_result[14];
measure q[15] -> m_result[15];
measure q[16] -> m_result[16];
measure q[17] -> m_result[17];
measure q[18] -> m_result[18];
measure q[19] -> m_result[19];
measure q[20] -> m_result[20];
measure q[21] -> m_result[21];
measure q[22] -> m_result[22];
measure q[23] -> m_result[23];
measure q[24] -> m_result[24];
measure q[25] -> m_result[25];
measure q[26] -> m_result[26];
measure q[27] -> m_result[27];
measure q[28] -> m_result[28];
measure q[29] -> m_result[29];
