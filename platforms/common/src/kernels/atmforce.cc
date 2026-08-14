KERNEL void hybridForce(int numParticles,
                        int paddedNumParticles,
                        GLOBAL mm_long* RESTRICT force,
                        GLOBAL mm_long* RESTRICT force0,
                        GLOBAL mm_long* RESTRICT force1,
                        GLOBAL mm_long* RESTRICT dforce0,
                        GLOBAL mm_long* RESTRICT dforce1,
                        GLOBAL int* RESTRICT invAtomOrder,
                        GLOBAL int* RESTRICT inner0InvAtomOrder,
                        GLOBAL int* RESTRICT inner1InvAtomOrder,
                        real dEdu0,
                        real dEdu1) {
    for (int i = GLOBAL_ID; i < numParticles; i += GLOBAL_SIZE) {
        int index = invAtomOrder[i];
        int index0 = inner0InvAtomOrder[i];
        int index1 = inner1InvAtomOrder[i];
        mm_long fx0 = force0[index0]+dforce0[index];
        mm_long fy0 = force0[index0+paddedNumParticles]+dforce0[index+paddedNumParticles];
        mm_long fz0 = force0[index0+paddedNumParticles*2]+dforce0[index+paddedNumParticles*2];
        mm_long fx1 = force1[index1]+dforce1[index];
        mm_long fy1 = force1[index1+paddedNumParticles]+dforce1[index+paddedNumParticles];
        mm_long fz1 = force1[index1+paddedNumParticles*2]+dforce1[index+paddedNumParticles*2];
        force[index]                      += (mm_long) (dEdu0*fx0 + dEdu1*fx1);
        force[index+paddedNumParticles]   += (mm_long) (dEdu0*fy0 + dEdu1*fy1);
        force[index+paddedNumParticles*2] += (mm_long) (dEdu0*fz0 + dEdu1*fz1);
    }
}

//reset variable displacement forces
KERNEL void resetDisplForce(int numParticles,
                            int paddedNumParticles,
                            GLOBAL mm_ulong* RESTRICT dforce0,
                            GLOBAL mm_ulong* RESTRICT dforce1) {
    mm_ulong zero = 0;
    for (int index = GLOBAL_ID; index < numParticles; index += GLOBAL_SIZE) {
        dforce0[index]                      = zero;
        dforce0[index+paddedNumParticles]   = zero;
        dforce0[index+paddedNumParticles*2] = zero;
        dforce1[index]                      = zero;
        dforce1[index+paddedNumParticles]   = zero;
        dforce1[index+paddedNumParticles*2] = zero;
    }
}

DEVICE inline real3 transformPosition(const real3* RESTRICT A1,
                                      const real3* RESTRICT B1,
                                      const real3* RESTRICT C1,
                                      const real3* RESTRICT A2,
                                      const real3* RESTRICT B2,
                                      const real3* RESTRICT C2,
                                      const real3* RESTRICT P1){
    
    // 1. Construct Frame 1 Orientation
    real3 v1 = *B1 - *A1;
    real3 u1 = *C1 - *A1;
    real  v1m = sqrt(dot(v1,v1));
    real3 z1 = v1/v1m;
    real  h1 = dot(u1, z1);
    real3 X1 = u1 - z1*h1;
    real  X1m = sqrt(dot(X1,X1));
    real3 x1 = X1/X1m;
    real3 y1 = cross(z1, x1);
    
    // 2. Project P1 into Local Frame 1 Space
    real3 dP1 = *P1 - *A1;
    real3 P_local = make_real3(dot(dP1, x1), dot(dP1, y1), dot(dP1, z1));
    
    // 3. Construct Frame 2 Orientation
    real3 v2 = *B2 - *A2;
    real3 u2 = *C2 - *A2;
    real  v2m = sqrt(dot(v2,v2));
    real3 z2 = v2/v2m;
    real  h2 = dot(u2, z2);
    real3 X2 = u2 - z2*h2;
    real  X2m = sqrt(dot(X2,X2));
    real3 x2 = X2/X2m;
    real3 y2 = cross(z2, x2);
    
    // 4. Reconstruct in Frame 2 Global Space
    real3 P2 = *A2 + x2*P_local.x + y2*P_local.y + z2*P_local.z;
    return P2;  
}


DEVICE inline void transformDerivatives(const real3* RESTRICT p_in,
                                        const real3* RESTRICT grad_in,
                                        const real3* RESTRICT a1,
                                        const real3* RESTRICT b1,
                                        const real3* RESTRICT c1,
                                        const real3* RESTRICT a2,
                                        const real3* RESTRICT b2,
                                        const real3* RESTRICT c2,
                                        real3* p_out,
                                        real3* grad_p_out,
                                        real3* grad_a1_out,
                                        real3* grad_b1_out,
                                        real3* grad_c1_out,
                                        real3* grad_a2_out,
                                        real3* grad_b2_out,
                                        real3* grad_c2_out
    ) {

    // Frame 1 Orientation Geometry
    real3 v1 = *b1 - *a1;
    real3 u1 = *c1 - *a1;
    real v1_norm = sqrt(dot(v1, v1));
    real3 z1 = v1 / v1_norm;
    real h1 = dot(u1, z1);
    real3 X1 = u1 - z1 * h1;
    real X1_norm = sqrt(dot(X1, X1));
    real3 x1 = X1 / X1_norm;
    real3 y1 = cross(z1, x1);
    
    // Project P1 into Local Frame 1 Space
    real3 dP1 = *p_in - *a1;
    real p_local_x = dot(dP1, x1);
    real p_local_y = dot(dP1, y1);
    real p_local_z = dot(dP1, z1);
    
    // Frame 2 Orientation Geometry
    real3 v2 = *b2 - *a2;
    real3 u2 = *c2 - *a2;
    real v2_norm = sqrt(dot(v2, v2));
    real3 z2 = v2 / v2_norm;
    real h2 = dot(u2, z2);
    real3 X2 = u2 - z2 * h2;
    real X2_norm = sqrt(dot(X2, X2));
    real3 x2 = X2 / X2_norm;
    real3 y2 = cross(z2, x2);
    
    // Compute Output Coordinates inside Global Frame 2
    (*p_out).x = (*a2).x + x2.x * p_local_x + y2.x * p_local_y + z2.x * p_local_z;
    (*p_out).y = (*a2).y + x2.y * p_local_x + y2.y * p_local_y + z2.y * p_local_z;
    (*p_out).z = (*a2).z + x2.z * p_local_x + y2.z * p_local_y + z2.z * p_local_z;

    // ==========================================================
    // 2. BACKWARD PASS (Adjoint Force Vector Propagation)
    // ==========================================================
    
    // Incoming adjoint from the potential energy gradient
    real3 F_p_out = *grad_in;

    // --- Backprop through Frame 2 Projection ---
    *grad_a2_out = F_p_out;
    
    real3 F_x2 = F_p_out * p_local_x;
    real3 F_y2 = F_p_out * p_local_y;
    real3 F_z2 = F_p_out * p_local_z;
    
    real F_p_local_x = dot(F_p_out, x2);
    real F_p_local_y = dot(F_p_out, y2);
    real F_p_local_z = dot(F_p_out, z2);

    // --- Backprop through Frame 2 Orientation ---
    // Adjoint of cross product: y2 = z2 cross x2
    real3 F_z2_mod = F_z2 + cross(x2, F_y2);
    real3 F_x2_mod = F_x2 + cross(F_y2, z2);

    // Adjoint of vector normalization: x2 = X2 / |X2|
    real3 F_X2 = (F_x2_mod - x2 * dot(F_x2_mod, x2)) / X2_norm;

    // Adjoint of orthogonal projection: X2 = u2 - z2 * h2
    real F_X2_dot_z2 = dot(F_X2, z2);
    real3 F_u2 = F_X2 - z2 * F_X2_dot_z2;
    F_z2_mod = F_z2_mod - F_X2 * h2 - u2 * F_X2_dot_z2;

    // Adjoint of vector normalization: z2 = v2 / |v2|
    real3 F_v2 = (F_z2_mod - z2 * dot(F_z2_mod, z2)) / v2_norm;

    // Track frame 2 anchor variables
    *grad_b2_out = F_v2;
    
    *grad_c2_out = F_u2;
    
    *grad_a2_out -= (F_v2 + F_u2);

    // --- Backprop through Local Frame Projection ---
    real3 F_dP1 = x1 * F_p_local_x + y1 * F_p_local_y + z1 * F_p_local_z;
    real3 F_x1  = dP1 * F_p_local_x;
    real3 F_y1  = dP1 * F_p_local_y;
    real3 F_z1  = dP1 * F_p_local_z;

    // dP1 = p_in - a1
    *grad_p_out  = F_dP1;
    
    // Explicitly initialize and map a1 gradients
    *grad_a1_out = -F_dP1;

    // --- Backprop through Frame 1 Orientation ---
    // Adjoint of cross product: y1 = z1 cross x1
    real3 F_z1_mod = F_z1 + cross(x1, F_y1);
    real3 F_x1_mod = F_x1 + cross(F_y1, z1);

    // Adjoint of vector normalization: x1 = X1 / |X1|
    real3 F_X1 = (F_x1_mod - x1 * dot(F_x1_mod, x1)) / X1_norm;

    // Adjoint of orthogonal projection: X1 = u1 - z1 * h1
    real F_X1_dot_z1 = dot(F_X1, z1);
    real3 F_u1 = F_X1 - z1 * F_X1_dot_z1;
    F_z1_mod = F_z1_mod - F_X1 * h1 - u1 * F_X1_dot_z1;

    // Adjoint of vector normalization: z1 = v1 / |v1|
    real3 F_v1 = (F_z1_mod - z1 * dot(F_z1_mod, z1)) / v1_norm;

    // Track frame 1 anchor variables
    *grad_b1_out = F_v1;
    
    *grad_c1_out = F_u1;
    
    *grad_a1_out -= (F_v1 + F_u1);
}


//add forces due to variable displacements
KERNEL void displForce(int numParticles,
                       int paddedNumParticles,
                       GLOBAL real4* RESTRICT posq,
                       GLOBAL mm_long* RESTRICT force0,
                       GLOBAL mm_long* RESTRICT force1,
                       GLOBAL mm_long* RESTRICT dforce0,
                       GLOBAL mm_long* RESTRICT dforce1,
                       GLOBAL int4* RESTRICT displParticles,
                       GLOBAL int4* RESTRICT displDestinationFrameParticles1,
                       GLOBAL int4* RESTRICT displSourceFrameParticles1,
                       GLOBAL int* RESTRICT atomOrder,
                       GLOBAL int* RESTRICT invAtomOrder,
                       GLOBAL int* RESTRICT inner0InvAtomOrder,
                       GLOBAL int* RESTRICT inner1InvAtomOrder) {
    GLOBAL mm_ulong* df0 = (GLOBAL mm_ulong*) dforce0;
    GLOBAL mm_ulong* df1 = (GLOBAL mm_ulong*) dforce1;
    for (int index = GLOBAL_ID; index < numParticles; index += GLOBAL_SIZE) {
        int atom = atomOrder[index];
        int pj1 = displParticles[atom].x;
        int pi1 = displParticles[atom].y;
        int pj0 = displParticles[atom].z;
        int pi0 = displParticles[atom].w;
        int index0 = inner0InvAtomOrder[atom];
        int index1 = inner1InvAtomOrder[atom];
        if (pj1 >= 0 && pi1 >= 0) {
            int j1 = invAtomOrder[pj1];
            int i1 = invAtomOrder[pi1];
            ATOMIC_ADD(&df1[j1], (mm_ulong) force1[index1]);
            ATOMIC_ADD(&df1[j1+paddedNumParticles], (mm_ulong) force1[index1+paddedNumParticles]);
            ATOMIC_ADD(&df1[j1+paddedNumParticles*2], (mm_ulong) force1[index1+paddedNumParticles*2]);
            ATOMIC_ADD(&df1[i1], (mm_ulong) -force1[index1]);
            ATOMIC_ADD(&df1[i1+paddedNumParticles], (mm_ulong) -force1[index1+paddedNumParticles]);
            ATOMIC_ADD(&df1[i1+paddedNumParticles*2],(mm_ulong)  -force1[index1+paddedNumParticles*2]);
        }
        if (pj0 >= 0 && pi0 >= 0) {
            int j0 = invAtomOrder[pj0];
            int i0 = invAtomOrder[pi0];
            ATOMIC_ADD(&df0[j0], (mm_ulong) force0[index0]);
            ATOMIC_ADD(&df0[j0+paddedNumParticles], (mm_ulong) force0[index0+paddedNumParticles]);
            ATOMIC_ADD(&df0[j0+paddedNumParticles*2],(mm_ulong)  force0[index0+paddedNumParticles*2]);
            ATOMIC_ADD(&df0[i0], (mm_ulong) -force0[index0]);
            ATOMIC_ADD(&df0[i0+paddedNumParticles], (mm_ulong) -force0[index0+paddedNumParticles]);
            ATOMIC_ADD(&df0[i0+paddedNumParticles*2], (mm_ulong) -force0[index0+paddedNumParticles*2]);
        }
        if (displDestinationFrameParticles1[atom].x >= 0) {
            //frame-based displacements
            real3 posP1 = make_real3((real) posq[index].x, (real) posq[index].y, (real) posq[index].z);
            int pAj1 = invAtomOrder[displDestinationFrameParticles1[atom].x];
            int pBj1 = invAtomOrder[displDestinationFrameParticles1[atom].y];
            int pCj1 = invAtomOrder[displDestinationFrameParticles1[atom].z];
            int pAi1 = invAtomOrder[displSourceFrameParticles1[atom].x];
            int pBi1 = invAtomOrder[displSourceFrameParticles1[atom].y];
            int pCi1 = invAtomOrder[displSourceFrameParticles1[atom].z];
            real3 posA2 = make_real3((real) posq[pAj1].x, (real) posq[pAj1].y, (real) posq[pAj1].z);
            real3 posB2 = make_real3((real) posq[pBj1].x, (real) posq[pBj1].y, (real) posq[pBj1].z);
            real3 posC2 = make_real3((real) posq[pCj1].x, (real) posq[pCj1].y, (real) posq[pCj1].z);
            real3 posA1 = make_real3((real) posq[pAi1].x, (real) posq[pAi1].y, (real) posq[pAi1].z);  
            real3 posB1 = make_real3((real) posq[pBi1].x, (real) posq[pBi1].y, (real) posq[pBi1].z);
            real3 posC1 = make_real3((real) posq[pCi1].x, (real) posq[pCi1].y, (real) posq[pCi1].z);
            real3 posP2;
            real  to_fixed = (real) 0x100000000LL; // 2^32
            real  to_real  = (1.0 / to_fixed);
            real3 force_p_in =  make_real3(to_real*force1[index1], to_real*force1[index1+paddedNumParticles], to_real*force1[index1+paddedNumParticles*2]);
            real3 force_p_out;
            real3 force_a1_out;
            real3 force_b1_out;
            real3 force_c1_out;
            real3 force_a2_out;
            real3 force_b2_out;
            real3 force_c2_out;
            transformDerivatives(&posP1, &force_p_in,
                                 &posA1, &posB1, &posC1,
                                 &posA2, &posB2, &posC2,
                                 &posP2,
                                 &force_p_out,
                                 &force_a1_out,
                                 &force_b1_out,
                                 &force_c1_out,
                                 &force_a2_out,
                                 &force_b2_out,
                                 &force_c2_out);
            
            ATOMIC_ADD(&df1[index],                      (mm_ulong) realToFixedPoint(force_p_out.x));
            ATOMIC_ADD(&df1[index+paddedNumParticles],   (mm_ulong) realToFixedPoint(force_p_out.y));
            ATOMIC_ADD(&df1[index+paddedNumParticles*2], (mm_ulong) realToFixedPoint(force_p_out.z));
            ATOMIC_ADD(&df1[index],                      (mm_ulong) realToFixedPoint(-force_p_in.x));
            ATOMIC_ADD(&df1[index+paddedNumParticles],   (mm_ulong) realToFixedPoint(-force_p_in.y));
            ATOMIC_ADD(&df1[index+paddedNumParticles*2], (mm_ulong) realToFixedPoint(-force_p_in.z));

            //DEBUG
            //printf("displForcesGPU: %d %f %f %f %f %f %f\n", atom, force_p_in.x, force_p_in.y, force_p_in.z, to_real*(mm_long)df1[index], to_real*(mm_long)df1[index+paddedNumParticles], to_real*(mm_long)df1[index+paddedNumParticles*2]);
            //printf("displForcesGPU: %d %f %f %f %f %f %f\n", atom, force_p_in.x, force_p_in.y, force_p_in.z, force_p_out.x-force_p_in.x, force_p_out.y-force_p_in.y, force_p_out.z-force_p_in.z);

            ATOMIC_ADD(&df1[pAj1],                      (mm_ulong) realToFixedPoint(force_a2_out.x));
            ATOMIC_ADD(&df1[pAj1+paddedNumParticles],   (mm_ulong) realToFixedPoint(force_a2_out.y));
            ATOMIC_ADD(&df1[pAj1+paddedNumParticles*2], (mm_ulong) realToFixedPoint(force_a2_out.z));
            
            ATOMIC_ADD(&df1[pBj1],                      (mm_ulong) realToFixedPoint(force_b2_out.x));
            ATOMIC_ADD(&df1[pBj1+paddedNumParticles],   (mm_ulong) realToFixedPoint(force_b2_out.y));
            ATOMIC_ADD(&df1[pBj1+paddedNumParticles*2], (mm_ulong) realToFixedPoint(force_b2_out.z));
            
            ATOMIC_ADD(&df1[pCj1],                      (mm_ulong) realToFixedPoint(force_c2_out.x));
            ATOMIC_ADD(&df1[pCj1+paddedNumParticles],   (mm_ulong) realToFixedPoint(force_c2_out.y));
            ATOMIC_ADD(&df1[pCj1+paddedNumParticles*2], (mm_ulong) realToFixedPoint(force_c2_out.z));

            ATOMIC_ADD(&df1[pAi1],                      (mm_ulong) realToFixedPoint(force_a1_out.x));
            ATOMIC_ADD(&df1[pAi1+paddedNumParticles],   (mm_ulong) realToFixedPoint(force_a1_out.y));
            ATOMIC_ADD(&df1[pAi1+paddedNumParticles*2], (mm_ulong) realToFixedPoint(force_a1_out.z));
            
            ATOMIC_ADD(&df1[pBi1],                      (mm_ulong) realToFixedPoint(force_b1_out.x));
            ATOMIC_ADD(&df1[pBi1+paddedNumParticles],   (mm_ulong) realToFixedPoint(force_b1_out.y));
            ATOMIC_ADD(&df1[pBi1+paddedNumParticles*2], (mm_ulong) realToFixedPoint(force_b1_out.z));
            
            ATOMIC_ADD(&df1[pCi1],                      (mm_ulong) realToFixedPoint(force_c1_out.x));
            ATOMIC_ADD(&df1[pCi1+paddedNumParticles],   (mm_ulong) realToFixedPoint(force_c1_out.y));
            ATOMIC_ADD(&df1[pCi1+paddedNumParticles*2], (mm_ulong) realToFixedPoint(force_c1_out.z));
        }
    }
}

KERNEL void copyState(int numParticles,
                      GLOBAL real4* RESTRICT posq,
                      GLOBAL real4* RESTRICT posq0,
                      GLOBAL real4* RESTRICT posq1,
                      GLOBAL real4* RESTRICT displacement0,
                      GLOBAL real4* RESTRICT displacement1,
                      GLOBAL int4* RESTRICT displParticles,
                      GLOBAL int4* RESTRICT displDestinationFrameParticles1,
                      GLOBAL int4* RESTRICT displSourceFrameParticles1,
                      GLOBAL int* RESTRICT atomOrder,
                      GLOBAL int* RESTRICT invAtomOrder,
                      GLOBAL int* RESTRICT inner0InvAtomOrder,
                      GLOBAL int* RESTRICT inner1InvAtomOrder
#ifdef USE_MIXED_PRECISION
                      ,
                      GLOBAL real4* RESTRICT posqCorrection,
                      GLOBAL real4* RESTRICT posq0Correction,
                      GLOBAL real4* RESTRICT posq1Correction
#endif
                    ) {

    for (int i = GLOBAL_ID; i < numParticles; i += GLOBAL_SIZE) {
        int atom = atomOrder[i]; //index in system

        //default fixed lab frame displacement
        real4 displ0 = displacement0[atom];
        real4 displ1 = displacement1[atom];
        //override with variable displacements if set
        int pj1 = displParticles[atom].x;
        int pi1 = displParticles[atom].y;
        int pj0 = displParticles[atom].z;
        int pi0 = displParticles[atom].w;
        if (pj1 >= 0 && pi1 >= 0) {
            // variable system coordinate displacements
            int indexj1 = invAtomOrder[pj1]; //index in reference context
            int indexi1 = invAtomOrder[pi1];
            displ1 = make_real4((real) posq[indexj1].x- posq[indexi1].x,
                                (real) posq[indexj1].y- posq[indexi1].y,
                                (real) posq[indexj1].z- posq[indexi1].z, (real) 0);
            if (pj0 >= 0 && pi0 >= 0) {
                int indexj0 = invAtomOrder[pj0];
                int indexi0 = invAtomOrder[pi0];
                displ0 = make_real4((real) posq[indexj0].x - posq[indexi0].x,
                                    (real) posq[indexj0].y - posq[indexi0].y,
                                    (real) posq[indexj0].z - posq[indexi0].z, (real) 0);
            }
            else {
                displ0 = make_real4(0);
            }
        }
        else if (displDestinationFrameParticles1[atom].x >= 0) {
            //frame-based displacements
            real3 posP1 = make_real3((real) posq[i].x, (real) posq[i].y, (real) posq[i].z);
            int pAj1 = invAtomOrder[displDestinationFrameParticles1[atom].x];
            int pBj1 = invAtomOrder[displDestinationFrameParticles1[atom].y];
            int pCj1 = invAtomOrder[displDestinationFrameParticles1[atom].z];
            int pAi1 = invAtomOrder[displSourceFrameParticles1[atom].x];
            int pBi1 = invAtomOrder[displSourceFrameParticles1[atom].y];
            int pCi1 = invAtomOrder[displSourceFrameParticles1[atom].z];
            real3 posA2 = make_real3((real) posq[pAj1].x, (real) posq[pAj1].y, (real) posq[pAj1].z);
            real3 posB2 = make_real3((real) posq[pBj1].x, (real) posq[pBj1].y, (real) posq[pBj1].z);
            real3 posC2 = make_real3((real) posq[pCj1].x, (real) posq[pCj1].y, (real) posq[pCj1].z);
            real3 posA1 = make_real3((real) posq[pAi1].x, (real) posq[pAi1].y, (real) posq[pAi1].z);  
            real3 posB1 = make_real3((real) posq[pBi1].x, (real) posq[pBi1].y, (real) posq[pBi1].z);
            real3 posC1 = make_real3((real) posq[pCi1].x, (real) posq[pCi1].y, (real) posq[pCi1].z);
            real3 posP2 = transformPosition(&posA1, &posB1, &posC1, &posA2, &posB2, &posC2, &posP1);
            displ1 = make_real4(posP2.x - posP1.x, posP2.y - posP1.y, posP2.z - posP1.z, 0);
            displ0 = make_real4(0); //TO-DO: implement displacement for state 0
            //DEBUG
            //printf("copyState(GPU): %d %f %f %f %f %f %f\n", atom,  posP1.x, posP1.y, posP1.z, posP2.x, posP2.y, posP2.z);
        }

        int index0 = inner0InvAtomOrder[atom];
        int index1 = inner1InvAtomOrder[atom];
        real4 p0 = posq[i] + make_real4((real) displ0.x, (real) displ0.y, (real) displ0.z, 0);
        real4 p1 = posq[i] + make_real4((real) displ1.x, (real) displ1.y, (real) displ1.z, 0);
        p0.w = posq0[i].w;
        p1.w = posq1[i].w;
        posq0[index0] = p0;
        posq1[index1] = p1;
#ifdef USE_MIXED_PRECISION
        posq0Correction[index0] = posqCorrection[i];
        posq1Correction[index1] = posqCorrection[i];
#endif
    }
}
