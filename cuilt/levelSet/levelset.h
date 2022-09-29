#ifndef _OPC_H_
#define _OPC_H_
#include <vector>
#include <string>
#include <fstream>
#include <cassert>
#include "shapes/design.h"
#include "lithosim/lithosimWrapper.h"
#include "lithosim/pvbandsim/kiss_fft.h"
#include "eval/epeChecker.h"

#define PI 3.141592 //3.14159265
#define ZERO_ERROR 0.000001
#define NONZERO(value) (value > ZERO_ERROR || value < -ZERO_ERROR)
#define CALL_MEMBER_FUNC(obj, ptrToMember) ((obj).*(ptrToMember))

const int LITHOSIM_OFFSET = 512;
//const int MASK_TILE_END_X = LITHOSIM_OFFSET + 1024;
//const int MASK_TILE_END_Y = LITHOSIM_OFFSET + 1024;
const int MASK_TILE_END_X = LITHOSIM_OFFSET + 1280;
const int MASK_TILE_END_Y = LITHOSIM_OFFSET + 1280;
const int NUM_MATRIX_TERM_ALLOCATE = 6;
const float NOMINAL_DOSE = 1;

//tunable =========
#define INITIAL_HAMMER_MASK
const int OPC_LENGTH_CORNER_RESHAPE = 10;

//#define ADD_SRAF
const int OPC_WIDTH_SRAF = 10;         //30;
const int OPC_SPACE_SRAF = 15;         //30;
const int OPC_SPACE_FORBID_SRAF = 100; //50;

const float OPC_KEEPOUT_REGION_RATIO = 0.25; //ratio of min feature width
const float OPC_RETARGET_INTENSITY_THRESHOLD = TARGET_INTENSITY / 3;
//because image is not binary, need stricter constraint
const float EPE_CONSTRAINT_INTERNAL = 0.5; //EPE_CONSTRAINT;// * 0.5;

//below parameters control gradient search
const float OPC_ERROR_THRESHOLD = 0;
//const int OPC_ITERATION_THRESHOLD = 20;
const int OPC_ITERATION_THRESHOLD = 20;
const float PHOTORISIST_SIGMOID_STEEPNESS = 50; //25;//80;
const float MASKRELAX_SIGMOID_STEEPNESS = 4;
const float EPERELAX_SIGMOID_STEEPNESS = 4; //2;
const float GRADIENT_DESCENT_STOP_CRITERIA = 0.015;
const float WEIGHT_REGULARIZATION = 0.025;
const float OBJ_FUNCTION_EPE_PENALTY = 1; // 1 means no penalty
const float WEIGHT_EPE_REGION = 0.5;      //non-epe region weight = 1
const float WEIGHT_PVBAND = 1;            //1;  //4
const float WEIGHT_EPE = 1250;            //5000;

//step size related
const float OPC_INITIAL_STEP_SIZE = 1; //4;//0.05;
const float OPC_JUMP_STEP_SIZE = OPC_INITIAL_STEP_SIZE / 2;
const float OPC_JUMP_STEP_THRESHOLD = OPC_INITIAL_STEP_SIZE / 3; ///4
const float GRADIENT_DESCENT_BETA = 0.75;                        //1 means constant step
const float GRADIENT_DESCENT_ALPHA = 0.5;
const int NUM_ITER_FIXED_STEP = 5; //10;
//====================

class OPC; //forward declaration
class Coordinate;

//function interface for callbacks
class OPCFunctionGroup
{
public:
    void (OPC::*initializeParams)();
    void (OPC::*calculateGradient)(int);
    void (OPC::*determineStepSize)(int); //can be removed
    void (OPC::*updateMask)();
    //float (OPC::*calculateObjValue) ();
    float (OPC::*calculatePxlObj)(int);
};

class OPC
{
public:
    typedef enum
    {
        HORIZONTAL,
        VERTICAL
    } orient_t;
    OPC(Design &design);
    ~OPC();

    void run();
    int getPvband() const { return m_pvband[m_numFinalIteration]; }
    int getNumEpe() const { return m_epeConvergence[m_numFinalIteration]; }

    static OPCFunctionGroup s_opcFuncGroup_v12;
    static OPCFunctionGroup s_opcFuncGroup_poonawala;

private:
    bool isPixelOn(int index) const { return (m_mask[index] >= MASK_PRINTABLE_THRESHOLD); }
    bool isPixelOn(float value) const { return value >= MASK_PRINTABLE_THRESHOLD; }
    bool isPixelIntensityOn(float value) const { return value >= TARGET_INTENSITY; }

    void rect2matrix(float *matrix, int originX, int originY);
    void matrix2rect(float *matrix, int originX, int originY);

    void initializeMask();
    void addSraf();
    void binaryMask();
    void binaryMask(float *bmask);
    void stochasticGD();
    int getIndexGeneral(int x, int y) const { return y * OPC_TILE_X + x; }
    int getIndex(int x, int y) const { return (y << 11) + x; }

    void initializeParams_cos();
    void initializeParams_sig();
    void calculateGradient_cos_poonawala(int numIteration = 0);
    void calculateGradient_sig_pvband8(int numIteration = 0);
    float calculateObjValue_pvband1();
    float calculateObjValue_pvband2();
    float calculateObjValue_pvband3();
    float calculateObjValue_pvband4();
    float calculateObjValue_pvband5();
    float calculateObjValue_pvband6();
    float calculateObjValue_target();
    float calculateObjValue();
    float calculatePxlObj_null(int index) { assert(0); }
    float calculatePxlObj_pvband(int index);
    float calculatePxlObj_p8(int index);
    bool exitOptIter(int numIteration);
    void determineStepSize_const(int numIteration);
    void determineStepSize_ta(int numIteration);
    void determineStepSize_backtrack(int numIteration);
    void updateMask_cos();
    void updateMask_sig();

    void updateConvergence(int numIteration);
    void dumpConvergence();
    //void getDiscretePenaly_sig(float *penalty);
    float getPxlDiscretePenaly_sig(int index);
    void drawHammer(int x, int y, int value);
    void drawSraf(int x, int y, int length, orient_t orient);
    bool isConvex(Coordinate *pre, Coordinate *cur, Coordinate *next);
    int calculatePatternArea();
    void findKeepOutRegion();
    void keepoutBridge_sig();
    void determineEPEWeight(int numIteration = 1);
    void determineEPEWeightBndy(int numIteration = 1);
    void keepBestResult(float curObj);
    void restoreBestResult(float curObj);
    void updateAvgIntensity(const float *intensity, int numIteration);
    void retarget(int numIteration);
    void dbgWriteBinaryImage(const char *fileName, float *image);

    //epe gradient
    int updateSampleEPE();
    float getSampleGradient(EpeSample *sample, int targetX, int targetY);
    void calcEpeGradient(kiss_fft_cpx *term1, kiss_fft_cpx *term2, kiss_fft_cpx *result);

    //matrix operation
    void subValue2Matrix(float value, float *source, float *target);
    void subMatrix2Value(float *source, float value, float *target);
    void addMatrix2Matrix(float *matrix1, float *matrix2, float *target);
    void addMatrix2Matrix(kiss_fft_cpx *matrix1, kiss_fft_cpx *matrix2, kiss_fft_cpx *target);
    void subMatrix2Matrix(float *matrix1, float *matrix2, float *target);
    void subMatrix2Matrix(kiss_fft_cpx *matrix1, kiss_fft_cpx *matrix2, kiss_fft_cpx *target);
    void multValue2Matrix(float value, float *matrix, float *target);
    void multMatrixE2E(float *matrix1, float *matrix2, float *target);
    void multMatrixE2E(float *matrix1, kiss_fft_cpx *matrix2, kiss_fft_cpx *target);

    //note overload priority larger than template
    void writeMatrix(std::string fileName, const kiss_fft_cpx *matrix);
    template <typename T>
    void writeMatrix(std::string fileName, const T *matrix)
    {
        std::fstream file(fileName.c_str(), std::fstream::out);
        for (int j = 0; j < OPC_TILE_Y; ++j)
        {
            for (int i = 0; i < OPC_TILE_X; ++i)
                file << (float)matrix[j * OPC_TILE_X + i] << " ";
            file << std::endl;
        }
        file.close();
    }

    OPCFunctionGroup m_opcFuncGroup;
    Design &m_design;
    LithosimWrapper m_lithosim;
    std::vector<EpeSample> *m_epeSamples;
    char *m_epeSamplePos;
    int m_minWidth;
    float m_minObjValue;

    float m_objConvergence[OPC_ITERATION_THRESHOLD + 1];
    int m_pvband[OPC_ITERATION_THRESHOLD + 1];
    int m_epeConvergence[OPC_ITERATION_THRESHOLD + 1];
    int m_scoreConvergence[OPC_ITERATION_THRESHOLD + 1];
    int m_numFinalIteration;
    float *m_params;
    float *m_stepSize;
    float *m_gradient;
    float *m_bestMask;
    float *m_mask;
    float *m_bmask;
    float *m_targetImage;
    float *m_image[3];
    float *m_avgIntensity;

    float *m_preObjValue;
    float *m_preGradient;
    float *m_preImage[NUM_PROCESS_CORNERS]; //delete
    float *m_oneMinusImage[NUM_PROCESS_CORNERS];
    float *m_diffImage;
    float *m_term[NUM_MATRIX_TERM_ALLOCATE];

    float *m_keepOutRegion;
    float *m_epeWeight;
    int *m_printHist;
    kiss_fft_cpx *m_cpxTerm[NUM_MATRIX_TERM_ALLOCATE];

    EpeChecker m_epeChecker;
};

#endif
