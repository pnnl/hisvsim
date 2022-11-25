OPENQASM 2.0;
include "qelib1.inc";
qreg qr[14];
creg cr[13];
rz(pi/2) qr[0];
sx qr[0];
rz(pi/2) qr[0];
rz(pi/2) qr[1];
sx qr[1];
rz(pi/2) qr[1];
rz(pi/2) qr[2];
sx qr[2];
rz(pi/2) qr[2];
rz(pi/2) qr[3];
sx qr[3];
rz(pi/2) qr[3];
rz(pi/2) qr[4];
sx qr[4];
rz(pi/2) qr[4];
rz(pi/2) qr[5];
sx qr[5];
rz(pi/2) qr[5];
rz(pi/2) qr[6];
sx qr[6];
rz(pi/2) qr[6];
rz(pi/2) qr[7];
sx qr[7];
rz(pi/2) qr[7];
rz(pi/2) qr[8];
sx qr[8];
rz(pi/2) qr[8];
rz(pi/2) qr[9];
sx qr[9];
rz(pi/2) qr[9];
rz(pi/2) qr[10];
sx qr[10];
rz(pi/2) qr[10];
rz(pi/2) qr[11];
sx qr[11];
rz(pi/2) qr[11];
rz(pi/2) qr[12];
sx qr[12];
rz(pi/2) qr[12];
rz(pi/2) qr[13];
sx qr[13];
rz(-pi/2) qr[13];
barrier qr[0],qr[1],qr[2],qr[3],qr[4],qr[5],qr[6],qr[7],qr[8],qr[9],qr[10],qr[11],qr[12],qr[13];
cx qr[0],qr[13];
cx qr[1],qr[13];
cx qr[2],qr[13];
cx qr[3],qr[13];
cx qr[4],qr[13];
cx qr[5],qr[13];
cx qr[6],qr[13];
cx qr[7],qr[13];
cx qr[8],qr[13];
cx qr[9],qr[13];
cx qr[10],qr[13];
cx qr[11],qr[13];
cx qr[12],qr[13];
barrier qr[0],qr[1],qr[2],qr[3],qr[4],qr[5],qr[6],qr[7],qr[8],qr[9],qr[10],qr[11],qr[12],qr[13];
rz(pi/2) qr[0];
sx qr[0];
rz(pi/2) qr[0];
rz(pi/2) qr[1];
sx qr[1];
rz(pi/2) qr[1];
rz(pi/2) qr[10];
sx qr[10];
rz(pi/2) qr[10];
rz(pi/2) qr[11];
sx qr[11];
rz(pi/2) qr[11];
rz(pi/2) qr[12];
sx qr[12];
rz(pi/2) qr[12];
rz(pi/2) qr[2];
sx qr[2];
rz(pi/2) qr[2];
rz(pi/2) qr[3];
sx qr[3];
rz(pi/2) qr[3];
rz(pi/2) qr[4];
sx qr[4];
rz(pi/2) qr[4];
rz(pi/2) qr[5];
sx qr[5];
rz(pi/2) qr[5];
rz(pi/2) qr[6];
sx qr[6];
rz(pi/2) qr[6];
rz(pi/2) qr[7];
sx qr[7];
rz(pi/2) qr[7];
rz(pi/2) qr[8];
sx qr[8];
rz(pi/2) qr[8];
rz(pi/2) qr[9];
sx qr[9];
rz(pi/2) qr[9];
