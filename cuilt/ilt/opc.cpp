#include <iostream>
#include <fstream>
#include <algorithm>
#include <vector>
#include <sstream>
#include <cstring>
#include <cmath>
#include <cstdlib>
#include <cfloat>
#include <climits>
#include <cassert>
#include <glog/logging.h>
#include "utils/debug.h"
#include "ilt/opc.h"
#include "shapes/design.h"
#include "shapes/shape.h"
#include "lithosim/pvbandsim/kiss_fft.h"
#include "proto/cuilt.pb.h"
#include "io/io.h"

//temp
extern float BACKGROUND_REAL;
extern float BACKGROUND_IMAG;
extern float MASK_REAL;
extern float MASK_IMAG;

OPCFunctionGroup OPC::s_opcFuncGroup_poonawala =
    {
        &OPC::initializeParams_cos,
        &OPC::calculateGradient_cos_poonawala,
        &OPC::determineStepSize_const,
        &OPC::updateMask_cos,
        &OPC::calculatePxlObj_null};

OPCFunctionGroup OPC::s_opcFuncGroup_v12 =
    {
        &OPC::initializeParams_sig,
        &OPC::calculateGradient_sig_pvband8,
        &OPC::determineStepSize_backtrack,
        &OPC::updateMask_sig,
        &OPC::calculatePxlObj_p8};

OPC::OPC(Design &design)
    : m_design(design),
      m_minWidth(1024), m_minObjValue(FLT_MAX), m_numFinalIteration(OPC_ITERATION_THRESHOLD)
{
    m_epeSamplePos = new char[OPC_TILE_SIZE];
    m_params = new float[OPC_TILE_SIZE];
    m_stepSize = new float[OPC_TILE_SIZE];
    m_gradient = new float[OPC_TILE_SIZE];
    m_preGradient = new float[OPC_TILE_SIZE];
    m_preObjValue = new float[OPC_TILE_SIZE];
    m_mask = new float[OPC_TILE_SIZE];
    //m_bmask = new float[OPC_TILE_SIZE];
    m_targetImage = new float[OPC_TILE_SIZE];
    memset(m_targetImage, 0, sizeof(float) * OPC_TILE_SIZE);
    m_diffImage = new float[OPC_TILE_SIZE];
    //  m_printHist = new int[OPC_TILE_SIZE];
    //  m_keepOutRegion = new float[OPC_TILE_SIZE];
    m_epeWeight = new float[OPC_TILE_SIZE];
    m_bestMask = new float[OPC_TILE_SIZE];
    m_avgIntensity = new float[OPC_TILE_SIZE];
    memset(m_avgIntensity, 0, sizeof(float) * OPC_TILE_SIZE);

    for (int i = 0; i < NUM_PROCESS_CORNERS; ++i)
    {
        m_image[i] = new float[OPC_TILE_SIZE];
        //    m_preImage[i] = new float[OPC_TILE_SIZE];
        m_oneMinusImage[i] = new float[OPC_TILE_SIZE];
    }
    for (int i = 0; i < NUM_MATRIX_TERM_ALLOCATE; ++i)
    {
        m_term[i] = new float[OPC_TILE_SIZE];
        m_cpxTerm[i] = new kiss_fft_cpx[OPC_TILE_SIZE];
    }

    m_opcFuncGroup = s_opcFuncGroup_v12;
    m_epeChecker.setDesign(m_design);
}

OPC::~OPC()
{
    delete m_epeSamplePos;
    delete m_params;
    delete m_stepSize;
    delete m_gradient;
    delete m_preGradient;
    delete m_preObjValue;
    delete m_mask;
    //delete m_bmask;
    delete m_targetImage;
    delete m_diffImage;
    //  delete m_printHist;
    //  delete m_keepOutRegion;
    delete m_epeWeight;
    delete m_bestMask;
    delete m_avgIntensity;

    for (int i = 0; i < NUM_PROCESS_CORNERS; ++i)
    {
        delete m_image[i];
        //    delete m_preImage[i];
        delete m_oneMinusImage[i];
    }
    for (int i = 0; i < NUM_MATRIX_TERM_ALLOCATE; ++i)
    {
        delete m_term[i];
        delete m_cpxTerm[i];
    }
}

void OPC::run()
{
    rect2matrix(m_targetImage, -LITHOSIM_OFFSET, -LITHOSIM_OFFSET);

    //tunable
    initializeMask();

#ifdef _DEBUG
    m_lithosim.writeImagePng((char *)"targetImg.png", m_targetImage, 1);
    m_lithosim.writeMaskPng((char *)"beforeopc.png", m_mask);
    LOG(INFO)<<"Total Pattern Area: " << calculatePatternArea();
#endif

    LOG(INFO) << "Starting OPC...";
    stochasticGD(); //do OPC
    LOG(INFO) << "Done OPC";

    //write mask rect back to design
    matrix2rect(m_mask, -LITHOSIM_OFFSET, -LITHOSIM_OFFSET);

#ifdef _DEBUG
    //  dbgWriteBinaryImage("bimg", m_image[2]);  //for tpl output
    m_lithosim.writeMaskPng((char *)"afteropc.png", m_mask);
#endif
}

//refer to lithosim.c::myDevLayerDraw
void OPC::rect2matrix(float *matrix, int originX, int originY)
{
    memset(matrix, 0, sizeof(float) * OPC_TILE_SIZE);
    const std::vector<Rect *> &rects = m_design.getRects();
    for (std::vector<Rect *>::const_iterator it = rects.begin(); it != rects.end(); ++it)
    {
        int llx = (*it)->ll().x - originX;
        int lly = (*it)->ll().y - originY;
        int urx = (*it)->ur().x - originX;
        int ury = (*it)->ur().y - originY;
        bool isOverlap = !(llx >= OPC_TILE_X || urx < 0 || lly >= OPC_TILE_Y || ury < 0);
        if (isOverlap)
        {
            for (int x = std::max(0, llx), xBound = std::min(OPC_TILE_X - 1, urx); x < xBound; ++x)
                for (int y = std::max(0, lly), yBound = std::min(OPC_TILE_Y - 1, ury); y < yBound; ++y)
                    matrix[getIndex(x, y)] = 1;
        }

        //obtain min feature width
        if (urx - llx > 22) //min:22nm
            m_minWidth = std::min(m_minWidth, urx - llx);
        if (ury - lly > 22)
            m_minWidth = std::min(m_minWidth, ury - lly);
    }
}

void OPC::matrix2rect(float *matrix, int originX, int originY)
{
    std::vector<Rect *> &rects = m_design.getMaskRects();
    //rectangle is formed row by row with height equals 1
    int start = -1;
    for (int y = LITHOSIM_OFFSET; y < MASK_TILE_END_Y; ++y)
        for (int x = LITHOSIM_OFFSET; x < MASK_TILE_END_X; ++x)
        {
            //end of a rectangle
            if (!isPixelOn(matrix[OPC_TILE_X * y + x]) || (x == MASK_TILE_END_X - 1)
                //|| !isPixelOn(matrix[OPC_TILE_X * (y + 1) + x])
                //this is to fix bug in input boundary condition, see rect2matrix
            )
            {
                if (start != -1 //found a rectangle
                )
                {
                    //note rect will be delete by caller
                    Rect *rect = new Rect(originX + start, originY + y, originX + x - 1, originY + y + 1);
                    rects.push_back(rect);
                    start = -1;
                }
            }
            else //start or continue extending rectangle
            {
                if (start == -1
                    //&& (isPixelOn(matrix[OPC_TILE_X * (y + 1) + x]))
                    //this is to fix bug in input boundary condition, see rect2matrix
                )
                    start = x;
            }
        }
}

//initial a mask solution
void OPC::initializeMask()
{
    int index;
    memset(m_mask, 0, sizeof(float) * OPC_TILE_SIZE);
    for (int y = LITHOSIM_OFFSET; y < MASK_TILE_END_Y; ++y)
        for (int x = LITHOSIM_OFFSET; x < MASK_TILE_END_X; ++x)
        {
            index = getIndex(x, y);
            m_mask[index] = m_targetImage[index];
            //m_mask[index] = rand()%2;
        }

        //  findKeepOutRegion();

#ifdef INITIAL_HAMMER_MASK
    //add hammerhead
    const std::vector<Rect *> &rects = m_design.getRects();
    for (int i = 0, len = m_design.getNumTrueRects(); i < len; ++i)
    {
        Rect *rect = rects[i];
        int llx = rect->ll().x + LITHOSIM_OFFSET;
        int lly = rect->ll().y + LITHOSIM_OFFSET;
        int urx = rect->ur().x + LITHOSIM_OFFSET;
        int ury = rect->ur().y + LITHOSIM_OFFSET;
        bool isOverlap = !(llx >= OPC_TILE_X || urx < 0 || lly >= OPC_TILE_Y || ury < 0);
        if (isOverlap)
        {
            drawHammer(llx, lly, 1);
            drawHammer(urx, lly, 1);
            drawHammer(urx, ury, 1);
            drawHammer(llx, ury, 1);
        }
    }

    const std::vector<Polygon *> &polygons = m_design.getPolygons();
    Polygon *polygon;
    Coordinate *pre, *cur, *next;
    int x, y;
    for (std::vector<Polygon *>::const_iterator it = polygons.begin();
         it != polygons.end(); ++it)
    {
        polygon = (*it);
        const std::vector<Coordinate *> &points = polygon->getPoints();
        int length = points.size();
        for (int i = 0; i < length; ++i)
        {
            cur = points[i];
            x = cur->x + LITHOSIM_OFFSET;
            y = cur->y + LITHOSIM_OFFSET;
            bool isOverlap = !(x >= OPC_TILE_X || x < 0 || y >= OPC_TILE_Y || y < 0);
            if (isOverlap)
            {
                pre = (i - 1 >= 0) ? points[i - 1] : points[length - 1];
                next = (i + 1 < length) ? points[i + 1] : points[0];
                if (isConvex(pre, cur, next))
                    drawHammer(x, y, 1);
                else //concave
                    drawHammer(x, y, 0);
            }
        }
    }
#endif

#ifdef ADD_SRAF
    addSraf();
#endif
}

void OPC::addSraf()
{
    const std::vector<Rect *> &rects = m_design.getRects();
    for (int i = 0, len = m_design.getNumTrueRects(); i < len; ++i)
    {
        Rect *rect = rects[i];
        int llx = rect->ll().x + LITHOSIM_OFFSET;
        int lly = rect->ll().y + LITHOSIM_OFFSET;
        int urx = rect->ur().x + LITHOSIM_OFFSET;
        int ury = rect->ur().y + LITHOSIM_OFFSET;
        bool isOverlap = !(llx >= OPC_TILE_X || urx < 0 || lly >= OPC_TILE_Y || ury < 0);
        if (isOverlap)
        {
            drawSraf(llx, lly, urx - llx, HORIZONTAL);
            drawSraf(llx, ury, urx - llx, HORIZONTAL);
            drawSraf(llx, lly, ury - lly, VERTICAL);
            drawSraf(urx, lly, ury - lly, VERTICAL);
        }
    }

    const std::vector<Polygon *> &polygons = m_design.getPolygons();
    Polygon *polygon;
    Coordinate *cur, *next;
    int x, y;
    for (std::vector<Polygon *>::const_iterator it = polygons.begin();
         it != polygons.end(); ++it)
    {
        polygon = (*it);
        const std::vector<Coordinate *> &points = polygon->getPoints();
        int length = points.size();
        for (int i = 0; i < length; ++i)
        {
            cur = points[i];
            x = cur->x + LITHOSIM_OFFSET;
            y = cur->y + LITHOSIM_OFFSET;
            bool isOverlap = !(x >= OPC_TILE_X || x < 0 || y >= OPC_TILE_Y || y < 0);
            if (isOverlap)
            {
                next = (i + 1 < length) ? points[i + 1] : points[0];
                if (cur->y == next->y) //HORIZONTAL
                    drawSraf(LITHOSIM_OFFSET + std::min(cur->x, next->x), LITHOSIM_OFFSET + cur->y,
                             abs(cur->x - next->x), HORIZONTAL);
                else //VERTICAL
                    drawSraf(LITHOSIM_OFFSET + cur->x, LITHOSIM_OFFSET + std::min(cur->y, next->y),
                             abs(cur->y - next->y), VERTICAL);
            }
        }
    }
}

void OPC::binaryMask()
{
    int index;
    for (int y = LITHOSIM_OFFSET; y < MASK_TILE_END_Y; ++y)
        for (int x = LITHOSIM_OFFSET; x < MASK_TILE_END_X; ++x)
        {
            index = getIndex(x, y);
            if (isPixelOn(index))
                m_mask[index] = 1;
            else
                m_mask[index] = 0;
        }
}

void OPC::binaryMask(float *bmask)
{
    int index;
    for (int y = LITHOSIM_OFFSET; y < MASK_TILE_END_Y; ++y)
        for (int x = LITHOSIM_OFFSET; x < MASK_TILE_END_X; ++x)
        {
            index = getIndex(x, y);
            if (isPixelOn(index))
                m_bmask[index] = 1;
            else
                m_bmask[index] = 0;
        }
}

//stochastic gradient descent
void OPC::stochasticGD()
{
    int numIteration = 0;
    int index;

    //initialize parameters
    CALL_MEMBER_FUNC(*this, m_opcFuncGroup.initializeParams)  
    ();
    CALL_MEMBER_FUNC(*this, m_opcFuncGroup.updateMask)
    ();
    determineEPEWeight();
    //  determineEPEWeightBndy(); //!!!!!!!

    //prepare for EPE calculation
    m_epeSamples = m_epeChecker.findSamplePoint();

    do
    {
        ++numIteration;
        LOG(INFO) << "Gradient search iteration " << numIteration;

        //    if (numIteration % 5 == 0) //!!!!!!
        //    {
        //      CALL_MEMBER_FUNC(*this, m_opcFuncGroup.initializeParams)();
        //      CALL_MEMBER_FUNC(*this, m_opcFuncGroup.updateMask)();
        //    }

        //determineEPEWeight(numIteration); //!!!!
        CALL_MEMBER_FUNC(*this, m_opcFuncGroup.calculateGradient)
        (numIteration);
        CALL_MEMBER_FUNC(*this, m_opcFuncGroup.determineStepSize)
        (numIteration);

        //now we have score of previous iteration
        updateConvergence(numIteration - 1);
        keepBestResult(m_scoreConvergence[numIteration - 1]);
#ifdef _DEBUG
        LOG(INFO) << "Objective value before updating: " << m_objConvergence[numIteration - 1];
        LOG(INFO) << "EPE count: " << m_epeConvergence[numIteration - 1];
        //    writeMatrix("mask", m_mask);
        //    writeMatrix("gradient1", m_gradient);
        //    writeMatrix("imageOuter1", m_image[0]);
        //    writeMatrix("imageInner1", m_image[1]);

        float maxg = 0, ming = 1024;
        for (int y = LITHOSIM_OFFSET; y < MASK_TILE_END_Y; ++y)
            for (int x = LITHOSIM_OFFSET; x < MASK_TILE_END_X; ++x)
            {
                maxg = std::max(maxg, m_gradient[getIndex(x, y)]);
                ming = std::min(ming, m_gradient[getIndex(x, y)]);
            }
        LOG(INFO) << "maxg: " << maxg << ", ming: " << ming;
#endif

        //update params
        for (int y = LITHOSIM_OFFSET; y < MASK_TILE_END_Y; ++y)
            for (int x = LITHOSIM_OFFSET; x < MASK_TILE_END_X; ++x)
            {
                index = getIndex(x, y);
                m_params[index] -= m_stepSize[index] * m_gradient[index];
            }

        CALL_MEMBER_FUNC(*this, m_opcFuncGroup.updateMask)
        ();
#ifdef _DEBUG
//    std::stringstream ss;
//    ss << numIteration;
//    writeMatrix("gradient" + ss.str(), m_gradient);
//writeMatrix("mask" + ss.str(), m_mask);
//writeMatrix("imageOuter" + ss.str(), m_image[0]);
//writeMatrix("imageInner" + ss.str(), m_image[1]);
#endif
        //keepoutBridge_sig();

    } while (!exitOptIter(numIteration));
    m_numFinalIteration = numIteration;

    //check result for the last iteration
    m_lithosim.simulateImageOpt(m_mask, m_image[0],
                                LithosimWrapper::LITHO_KERNEL_FOCUS, MAX_DOSE); //outer corner: focus, +2% dose
    m_lithosim.simulateImageOpt(m_mask, m_image[1],
                                LithosimWrapper::LITHO_KERNEL_DEFOCUS, MIN_DOSE); //inner corner: defocus, -2% dose
    m_lithosim.simulateImageOpt(m_mask, m_image[2],
                                LithosimWrapper::LITHO_KERNEL_FOCUS, NOMINAL_DOSE); //nominal: focus, dose 1

    m_pvband[numIteration] = m_lithosim.calculatePvband(m_image[1], m_image[0]);
    m_epeConvergence[numIteration] = m_epeChecker.run(m_image[2]);
    for (int i = 0; i < OPC_TILE_SIZE; ++i)
    {
        m_image[0][i] = 1.0 /
                        (1.0 + exp(-PHOTORISIST_SIGMOID_STEEPNESS * (m_image[0][i] - TARGET_INTENSITY)));
        m_image[1][i] = 1.0 /
                        (1.0 + exp(-PHOTORISIST_SIGMOID_STEEPNESS * (m_image[1][i] - TARGET_INTENSITY)));
        m_image[2][i] = 1.0 /
                        (1.0 + exp(-PHOTORISIST_SIGMOID_STEEPNESS * (m_image[2][i] - TARGET_INTENSITY)));
    }
    updateConvergence(numIteration);
    LOG(INFO) << "Final objective value: " << m_objConvergence[numIteration];
    restoreBestResult(m_scoreConvergence[numIteration]);

#ifdef _DEBUG
    //dump info for the final mask
    binaryMask();
    //outer corner: focus, +2% dose
    m_lithosim.simulateImageOpt(m_mask, m_image[0],
                                LithosimWrapper::LITHO_KERNEL_FOCUS, MAX_DOSE, NUM_LITHO_KERNELS);
    //inner corner: defocus, -2% dose
    m_lithosim.simulateImageOpt(m_mask, m_image[1],
                                LithosimWrapper::LITHO_KERNEL_DEFOCUS, MIN_DOSE, NUM_LITHO_KERNELS);
    //nominal: focus, dose 1
    m_lithosim.simulateImageOpt(m_mask, m_image[2],
                                LithosimWrapper::LITHO_KERNEL_FOCUS, NOMINAL_DOSE, NUM_LITHO_KERNELS);

    m_lithosim.writeImageEpePng((char *)"imgNominalF.png", m_image[2], NOMINAL_DOSE,
                                m_design.getLayoutFile());
    m_lithosim.writeImageEpePng((char *)"imgOuterF.png", m_image[0], MAX_DOSE);
    m_lithosim.writeImageEpePng((char *)"imgInnerF.png", m_image[1], MIN_DOSE);
    m_lithosim.writePvband((char *)"pvband.png", m_image[1], m_image[0]);
    m_pvband[numIteration] = m_lithosim.calculatePvband(m_image[1], m_image[0]);

    //check EPE based on nominal process condition
    m_epeConvergence[numIteration] = m_epeChecker.run(m_image[2]);

    for (int i = 0; i < OPC_TILE_SIZE; ++i)
    {
        m_image[0][i] = 1.0 /
                        (1.0 + exp(-PHOTORISIST_SIGMOID_STEEPNESS * (m_image[0][i] - TARGET_INTENSITY)));
        m_image[1][i] = 1.0 /
                        (1.0 + exp(-PHOTORISIST_SIGMOID_STEEPNESS * (m_image[1][i] - TARGET_INTENSITY)));
        m_image[2][i] = 1.0 /
                        (1.0 + exp(-PHOTORISIST_SIGMOID_STEEPNESS * (m_image[2][i] - TARGET_INTENSITY)));
    }

    updateConvergence(numIteration);
    LOG(INFO) << "Final objective value: " << m_objConvergence[numIteration];
#endif
    dumpConvergence();
}

//tunable
bool OPC::exitOptIter(int numIteration)
{
    bool meetMaxIter = (numIteration >= OPC_ITERATION_THRESHOLD);
#define ENABLE_MSE
#ifdef ENABLE_MSE
    if (meetMaxIter)
        return true;
#endif

    float mse = 0;
    for (int i = 0; i < OPC_TILE_SIZE; ++i)
        mse += m_gradient[i] * m_gradient[i];
    mse = sqrt(mse / OPC_TILE_SIZE);
    LOG(INFO) << "gradient mse: " << mse;
#ifdef ENABLE_MSE
    return (mse < GRADIENT_DESCENT_STOP_CRITERIA);
#else
    return meetMaxIter;
#endif

    //return meetMaxIter || (error >= preError); //stop at local optima
    //return meetMaxIter || (opcError > OPC_ERROR_THRESHOLD);
}

//must have m_image calculated before calling this function
float OPC::calculateObjValue_pvband1()
{
    subMatrix2Matrix(m_image[0], m_image[1], m_diffImage); //Z1-Z2
    float sumSquare = 0;
    int index;
    for (int y = LITHOSIM_OFFSET; y < MASK_TILE_END_Y; ++y)
        for (int x = LITHOSIM_OFFSET; x < MASK_TILE_END_X; ++x)
        {
            index = getIndex(x, y);
            sumSquare += m_diffImage[index] * m_diffImage[index];
        }
    return sumSquare;
}

//must have m_image calculated before calling this function
float OPC::calculateObjValue_pvband2()
{
    subMatrix2Matrix(m_image[0], m_targetImage, m_term[0]); //Z^-Z1
    subMatrix2Matrix(m_image[1], m_targetImage, m_term[1]); //Z^-Z2
    float sumSquare = 0;
    int index;
    for (int y = LITHOSIM_OFFSET; y < MASK_TILE_END_Y; ++y)
        for (int x = LITHOSIM_OFFSET; x < MASK_TILE_END_X; ++x)
        {
            index = getIndex(x, y);
            sumSquare += m_epeWeight[index] *
                         (m_term[0][index] * m_term[0][index] + m_term[1][index] * m_term[1][index]);
        }
    return sumSquare;
}

//must have m_image calculated before calling this function
float OPC::calculateObjValue_pvband3()
{
    //subMatrix2Matrix(m_image[0], m_targetImage, m_term[0]); //Z^-Z1
    subMatrix2Matrix(m_image[1], m_targetImage, m_term[1]); //Z^-Z2
    subMatrix2Matrix(m_image[0], m_image[1], m_diffImage);  //Z1-Z2
    float sumSquare = 0;
    int index;
    for (int y = LITHOSIM_OFFSET; y < MASK_TILE_END_Y; ++y)
        for (int x = LITHOSIM_OFFSET; x < MASK_TILE_END_X; ++x)
        {
            index = getIndex(x, y);
            //    sumSquare += m_term[0][index] * m_term[0][index] + m_term[1][index] * m_term[1][index]
            //      sumSquare += OBJ_FUNCTION_EPE_PENALTY * (m_term[1][index] * m_term[1][index])
            //        + m_diffImage[index] * m_diffImage[index];
            sumSquare += m_epeWeight[index] * (m_term[1][index] * m_term[1][index]) + m_diffImage[index] * m_diffImage[index];
        }
    return sumSquare;
}

//must have m_image calculated before calling this function
//float
//OPC::calculateObjValue_pvband4()
//{
//  subMatrix2Matrix(m_image[2], m_targetImage, m_term[1]); //Z^-Z0
//  subMatrix2Matrix(m_image[0], m_image[1], m_diffImage); //Z1-Z2
//  getDiscretePenaly_sig(m_term[0]);
//  float sumSquare = 0;
//  int index;
//  for (int y = LITHOSIM_OFFSET; y < MASK_TILE_END_Y; ++y)
//    for (int x = LITHOSIM_OFFSET; x < MASK_TILE_END_X; ++x)
//    {
//      index = getIndex(x, y);
//      //sumSquare += OBJ_FUNCTION_EPE_PENALTY * (m_term[1][index] * m_term[1][index])
//      sumSquare += m_epeWeight[index] * (m_term[1][index] * m_term[1][index])
//        + m_diffImage[index] * m_diffImage[index]
//        + WEIGHT_REGULARIZATION * m_term[0][index];
//    }
//  return sumSquare;
//}

//must have m_image calculated before calling this function
float OPC::calculateObjValue_pvband5()
{
    subMatrix2Matrix(m_image[2], m_targetImage, m_term[0]); //Z^-Z0
    subMatrix2Matrix(m_image[1], m_targetImage, m_term[1]); //Z^-Z2
    float sumSquare = 0;
    int index;
    for (int y = LITHOSIM_OFFSET; y < MASK_TILE_END_Y; ++y)
        for (int x = LITHOSIM_OFFSET; x < MASK_TILE_END_X; ++x)
        {
            index = getIndex(x, y);
            sumSquare += m_epeWeight[index] *
                         (m_term[0][index] * m_term[0][index] + m_term[1][index] * m_term[1][index]);
        }
    return sumSquare;
}

//must have m_image calculated before calling this function
float OPC::calculateObjValue_target()
{
    subMatrix2Matrix(m_image[0], m_targetImage, m_diffImage); //1-Z1
    float sumSquare = 0;
    int index;
    for (int y = LITHOSIM_OFFSET; y < MASK_TILE_END_Y; ++y)
        for (int x = LITHOSIM_OFFSET; x < MASK_TILE_END_X; ++x)
        {
            index = getIndex(x, y);
            sumSquare += m_diffImage[index] * m_diffImage[index];
        }
    return sumSquare;
}

//must have m_image calculated before calling this function
float OPC::calculateObjValue()
{
    float sum = 0;
    int index;
    for (int y = LITHOSIM_OFFSET; y < MASK_TILE_END_Y; ++y)
        for (int x = LITHOSIM_OFFSET; x < MASK_TILE_END_X; ++x)
        {
            index = getIndex(x, y);
            sum += CALL_MEMBER_FUNC(*this, m_opcFuncGroup.calculatePxlObj)(index);
        }
    return sum;
}

void OPC::updateConvergence(int numIteration)
{
    m_scoreConvergence[numIteration] = 5000 * m_epeConvergence[numIteration] + 4 *
                                                                                   m_pvband[numIteration];
    m_objConvergence[numIteration] = calculateObjValue();
}

float OPC::calculatePxlObj_pvband(int index)
{
    float diffImage = m_image[0][index] - m_image[1][index]; //Z1-Z2
    return diffImage * diffImage;
}

float OPC::calculatePxlObj_p8(int index)
{
    float diffTarget = m_image[2][index] - m_targetImage[index]; //Z^-Z0
    //float diffImage = m_image[0][index] - m_targetImage[index]; //Z^-Z2
    float diffImage = m_image[0][index] - m_image[1][index]; //Z1^-Z2
    float discretePenalty = getPxlDiscretePenaly_sig(index);
    return m_epeWeight[index] * diffTarget * diffTarget * diffTarget * diffTarget + diffImage * diffImage + WEIGHT_REGULARIZATION * discretePenalty;
}

float OPC::getPxlDiscretePenaly_sig(int index)
{
    return WEIGHT_REGULARIZATION * (-8 * m_mask[index] + 4);
}

//obj=(Z0-Z^)^4+(Z^-Z2)^2
void OPC::calculateGradient_sig_pvband8(int numIteration)
{
    m_lithosim.simulateImageOpt(m_mask, m_image[0], //M -> Znom
                                LithosimWrapper::LITHO_KERNEL_FOCUS, MAX_DOSE); //outer corner: focus, +2% dose
    m_lithosim.simulateImageOpt(m_mask, m_image[1],
                                LithosimWrapper::LITHO_KERNEL_DEFOCUS, MIN_DOSE); //inner corner: defocus, -2% dose
    m_lithosim.simulateImageOpt(m_mask, m_image[2],
                                LithosimWrapper::LITHO_KERNEL_FOCUS, NOMINAL_DOSE); //nominal: focus, dose 1

    //apply retargeting !!!!!!
    //  updateAvgIntensity(m_image[2], numIteration);
    //  if (numIteration % 5 == 0)
    //    retarget(numIteration);

    //to update score
    m_pvband[numIteration - 1] = m_lithosim.calculatePvband(m_image[1], m_image[0]); //inner, outer
    m_epeConvergence[numIteration - 1] = m_epeChecker.run(m_image[2]);

#ifdef _DEBUG
    std::stringstream ss;
    ss << numIteration;
    std::string fn1 = "imgOuter" + ss.str() + ".png";
    std::string fn2 = "imgInner" + ss.str() + ".png";
    m_lithosim.writeImagePng((char *)fn1.c_str(), m_image[0], MAX_DOSE);
    m_lithosim.writeImagePng((char *)fn2.c_str(), m_image[1], MIN_DOSE);
    LOG(INFO) << "pvband before updating: " << m_pvband[numIteration - 1];
#endif

    //apply sigmoid function
    for (int i = 0; i < OPC_TILE_SIZE; ++i)
    {
        m_image[0][i] = 1.0 /
                        (1.0 + exp(-PHOTORISIST_SIGMOID_STEEPNESS * (m_image[0][i] - TARGET_INTENSITY)));
        m_image[1][i] = 1.0 /
                        (1.0 + exp(-PHOTORISIST_SIGMOID_STEEPNESS * (m_image[1][i] - TARGET_INTENSITY)));
        m_image[2][i] = 1.0 /
                        (1.0 + exp(-PHOTORISIST_SIGMOID_STEEPNESS * (m_image[2][i] - TARGET_INTENSITY)));
    }

    //calculate d (Z0-Z^)^4 ==========================
    subMatrix2Matrix(m_image[2], m_targetImage, m_diffImage); //Z-Z^
    multMatrixE2E(m_diffImage, m_diffImage, m_term[0]);       //(Z-Z^)^2
    multMatrixE2E(m_diffImage, m_term[0], m_term[0]);         //(Z-Z^)^3
    subValue2Matrix(1, m_image[2], m_oneMinusImage[0]);       //1-Z
    multMatrixE2E(m_term[0], m_image[2], m_term[0]);          //(Z-Z^)^3 Z
    multMatrixE2E(m_term[0], m_oneMinusImage[0], m_term[0]);  //(Z-Z^)^3 Z(1-Z)

    //H*(t0 M*H')
    m_lithosim.convolveKernel(m_mask, m_cpxTerm[0],
                              LithosimWrapper::LITHO_KERNEL_FOCUS_CT, MAX_DOSE);
    multMatrixE2E(m_term[0], m_cpxTerm[0], m_cpxTerm[0]);
    m_lithosim.convolveKernel(m_cpxTerm[0], m_cpxTerm[1],
                              LithosimWrapper::LITHO_KERNEL_FOCUS, MAX_DOSE);

    //H'*(t0 M*H)
    m_lithosim.convolveKernel(m_mask, m_cpxTerm[0],
                              LithosimWrapper::LITHO_KERNEL_FOCUS, MAX_DOSE);
    multMatrixE2E(m_term[0], m_cpxTerm[0], m_cpxTerm[0]);
    m_lithosim.convolveKernel(m_cpxTerm[0], m_cpxTerm[2],
                              LithosimWrapper::LITHO_KERNEL_FOCUS_CT, MAX_DOSE);

    addMatrix2Matrix(m_cpxTerm[1], m_cpxTerm[2], m_cpxTerm[0]);

    int index;
    for (int y = LITHOSIM_OFFSET; y < MASK_TILE_END_Y; ++y)
        for (int x = LITHOSIM_OFFSET; x < MASK_TILE_END_X; ++x)
        {
            index = getIndex(x, y);
            m_gradient[index] = m_epeWeight[index] * m_cpxTerm[0][index].r; //ignore imageniry
        }

    //calculate d (Z^-Z2)^2 ==========================
    subMatrix2Matrix(m_image[1], m_targetImage, m_diffImage); //Z-Z^
    subValue2Matrix(1, m_image[1], m_oneMinusImage[0]);       //1-Z
    multMatrixE2E(m_diffImage, m_image[1], m_term[0]);        //(Z-Z^)Z
    multMatrixE2E(m_term[0], m_oneMinusImage[0], m_term[0]);  //(Z-Z^)I(1-Z)

    //H*(t0 M*H')
    m_lithosim.convolveKernel(m_mask, m_cpxTerm[0],
                              LithosimWrapper::LITHO_KERNEL_DEFOCUS_CT, MIN_DOSE);
    multMatrixE2E(m_term[0], m_cpxTerm[0], m_cpxTerm[0]);
    m_lithosim.convolveKernel(m_cpxTerm[0], m_cpxTerm[1],
                              LithosimWrapper::LITHO_KERNEL_DEFOCUS, MIN_DOSE);

    //H'*(t0 M*H)
    m_lithosim.convolveKernel(m_mask, m_cpxTerm[0],
                              LithosimWrapper::LITHO_KERNEL_DEFOCUS, MIN_DOSE);
    multMatrixE2E(m_term[0], m_cpxTerm[0], m_cpxTerm[0]);
    m_lithosim.convolveKernel(m_cpxTerm[0], m_cpxTerm[2],
                              LithosimWrapper::LITHO_KERNEL_DEFOCUS_CT, MIN_DOSE);

    addMatrix2Matrix(m_cpxTerm[1], m_cpxTerm[2], m_cpxTerm[0]);

    //mult -2alpha*A and M (1-M), and sum up
    float constant1 = 4 * PHOTORISIST_SIGMOID_STEEPNESS * MASKRELAX_SIGMOID_STEEPNESS;
    float constant2 = WEIGHT_PVBAND * 2 * PHOTORISIST_SIGMOID_STEEPNESS * MASKRELAX_SIGMOID_STEEPNESS;
    for (int y = LITHOSIM_OFFSET; y < MASK_TILE_END_Y; ++y)
        for (int x = LITHOSIM_OFFSET; x < MASK_TILE_END_X; ++x)
        {
            index = getIndex(x, y);
            m_gradient[index] = (constant1 * m_gradient[index] + constant2 * m_cpxTerm[0][index].r + MASKRELAX_SIGMOID_STEEPNESS * getPxlDiscretePenaly_sig(index)) * m_mask[index] * (1.0 - m_mask[index]);
        }

    LOG(INFO) << "Done calculate gradient. ";
}

//obj=(Z^-Z1)^2
void OPC::calculateGradient_cos_poonawala(int numIteration)
{
    m_lithosim.simulateImageOpt(m_mask, m_image[0],
                                LithosimWrapper::LITHO_KERNEL_FOCUS, MAX_DOSE); //outer corner: focus, +2% dose
                                                                                //  m_lithosim.simulateImageOpt(m_mask,m_image[1],
                                                                                //      LithosimWrapper::LITHO_KERNEL_DEFOCUS, MIN_DOSE);  //inner corner: defocus, -2% dose
    //apply sigmoid function
    for (int i = 0; i < OPC_TILE_SIZE; ++i)
    {
        m_image[0][i] = 1.0 /
                        (1.0 + exp(-PHOTORISIST_SIGMOID_STEEPNESS * (m_image[0][i] - TARGET_INTENSITY)));
        //    m_image[1][i] = 1.0 /
        //      (1.0 + exp(-PHOTORISIST_SIGMOID_STEEPNESS * (m_image[1][i] - TARGET_INTENSITY)));
    }

    subMatrix2Matrix(m_image[0], m_targetImage, m_diffImage); //I-I^
    subValue2Matrix(1, m_image[0], m_oneMinusImage[0]);       //1-I
    multMatrixE2E(m_diffImage, m_image[0], m_term[0]);        //(I-I^)I
    multMatrixE2E(m_term[0], m_oneMinusImage[0], m_term[0]);  //(I-I^)I(1-I)

    //H1'*t0
    m_lithosim.convolveKernel(m_term[0], m_cpxTerm[0],
                              LithosimWrapper::LITHO_KERNEL_FOCUS_CT, MAX_DOSE);

    //  //H2'*t0
    //  m_lithosim.convolveKernel(m_term[0], m_cpxTerm[1],
    //      LithosimWrapper::LITHO_KERNEL_DEFOCUS_CT, MIN_DOSE);

    //mult a and sin theta
    int index;
    for (int y = LITHOSIM_OFFSET; y < MASK_TILE_END_Y; ++y)
        for (int x = LITHOSIM_OFFSET; x < MASK_TILE_END_X; ++x)
        {
            index = getIndex(x, y);
            m_gradient[index] = 90 * (m_cpxTerm[0][index].r + m_cpxTerm[1][index].r) * sin(m_params[index]); //ignore imageniry
        }
    LOG(INFO) << "Done calculate gradient. ";
}

void OPC::initializeParams_cos()
{
    //  float on = 0.1 * PI;
    //  float off = 0.9 * PI;
    int index;
    for (int y = LITHOSIM_OFFSET; y < MASK_TILE_END_Y; ++y)
        for (int x = LITHOSIM_OFFSET; x < MASK_TILE_END_X; ++x)
        {
            index = getIndex(x, y);
            if (isPixelOn(index))
                m_params[index] = 0; //cos0=1
            //m_params[index] = on;
            else
                m_params[index] = PI; //cos pi=-1
                                      //m_params[index] = off;
        }
}

void OPC::initializeParams_sig()
{
    int index;
    for (int y = LITHOSIM_OFFSET; y < MASK_TILE_END_Y; ++y)
        for (int x = LITHOSIM_OFFSET; x < MASK_TILE_END_X; ++x)
        {
            index = getIndex(x, y);
            if (isPixelOn(index))
                m_params[index] = 1; //1/(1+e^(-A*1))=1
            else
                m_params[index] = -1; //1/(1+e^(-A*-1))=0
        }
}

void OPC::determineStepSize_const(int numIteration)
{
    if (numIteration == 1)
    {
        int index;
        for (int y = LITHOSIM_OFFSET; y < MASK_TILE_END_Y; ++y)
            for (int x = LITHOSIM_OFFSET; x < MASK_TILE_END_X; ++x)
            {
                index = getIndex(x, y);
                m_stepSize[index] = OPC_INITIAL_STEP_SIZE;
            }
    }
}

//threshold acceptence+history
void OPC::determineStepSize_ta(int numIteration)
{
    int index;
    if (numIteration == 1) //initialize at fist iteration
    {
        for (int y = LITHOSIM_OFFSET; y < MASK_TILE_END_Y; ++y)
            for (int x = LITHOSIM_OFFSET; x < MASK_TILE_END_X; ++x)
            {
                index = getIndex(x, y);
                m_stepSize[index] = OPC_INITIAL_STEP_SIZE;

                //update history
                if (isPixelOn(index))
                    ++m_printHist[index];
            }
    }
    else if (numIteration > NUM_ITER_FIXED_STEP)
    {
        int count = 0;
        float curPixel, nextPixel;
        for (int y = LITHOSIM_OFFSET; y < MASK_TILE_END_Y; ++y)
            for (int x = LITHOSIM_OFFSET; x < MASK_TILE_END_X; ++x)
            {
                index = getIndex(x, y);
                if (isPixelOn(index))
                    ++m_printHist[index];

                //try to blur the solution
                curPixel = m_mask[index];
                nextPixel = 1.0 / (1.0 + exp(-MASKRELAX_SIGMOID_STEEPNESS *
                                                 m_params[index] -
                                             m_stepSize[index] * m_gradient[index]));
                if (isPixelOn(curPixel) && !isPixelOn(nextPixel)) //1->0
                                                                  //        if (m_gradient[index] < 0)  //toward 0
                {
                    //move to opposite direction if higher pixelOn probability
                    if ((rand() % 100) / 100.0 >= (float)m_printHist[index] / numIteration)
                    {
                        m_stepSize[index] = -OPC_INITIAL_STEP_SIZE;
                        ++count;
                    }
                    else
                        m_stepSize[index] = OPC_INITIAL_STEP_SIZE;
                }
                else if (!isPixelOn(curPixel) && isPixelOn(nextPixel)) //0->1
                                                                       //        else  //toward 1
                {
                    //move to opposite direction if lower pixelOn probability
                    if ((rand() % 100) / 100.0 < (float)m_printHist[index] / numIteration)
                    {
                        m_stepSize[index] = -OPC_INITIAL_STEP_SIZE;
                        ++count;
                    }
                    else
                        m_stepSize[index] = OPC_INITIAL_STEP_SIZE;
                }
                //        else  //no flip
                //            m_stepSize[index] = OPC_INITIAL_STEP_SIZE;
            }
        LOG(INFO) << count << " gradient updated to opposite direction.";
    }
    else
    {
        //accumlate history from mask
        for (int y = LITHOSIM_OFFSET; y < MASK_TILE_END_Y; ++y)
            for (int x = LITHOSIM_OFFSET; x < MASK_TILE_END_X; ++x)
            {
                index = getIndex(x, y);
                if (isPixelOn(index))
                    ++m_printHist[index];
            }
    }
}

void OPC::determineStepSize_backtrack(int numIteration)
{
    int index;
    if (numIteration == 1) //initialize at fist iteration
                           //  if (numIteration == 1 || (numIteration % 5) == 0)
    {
        for (int y = LITHOSIM_OFFSET; y < MASK_TILE_END_Y; ++y)
            for (int x = LITHOSIM_OFFSET; x < MASK_TILE_END_X; ++x)
            {
                index = getIndex(x, y);
                m_stepSize[index] = OPC_INITIAL_STEP_SIZE;
            }

        //calculate objvalue of each pixel
        for (int y = LITHOSIM_OFFSET; y < MASK_TILE_END_Y; ++y)
            for (int x = LITHOSIM_OFFSET; x < MASK_TILE_END_X; ++x)
            {
                index = getIndex(x, y);
                m_preObjValue[index] =
                    CALL_MEMBER_FUNC(*this, m_opcFuncGroup.calculatePxlObj)(index);
            }
    }
    else
    {
        int countJump = 0;
        int countReduce = 0;
        float curObj;
        for (int y = LITHOSIM_OFFSET; y < MASK_TILE_END_Y; ++y)
            for (int x = LITHOSIM_OFFSET; x < MASK_TILE_END_X; ++x)
            {
                index = getIndex(x, y);
                if (m_stepSize[index] < OPC_JUMP_STEP_THRESHOLD) //jump out of local optima
                {
                    m_stepSize[index] = OPC_JUMP_STEP_SIZE;
                    ++countJump;
                }
                else
                {
                    curObj = CALL_MEMBER_FUNC(*this, m_opcFuncGroup.calculatePxlObj)(index);
                    if (m_preObjValue[index] - curObj <
                        GRADIENT_DESCENT_ALPHA * m_stepSize[index] * m_preGradient[index] * m_preGradient[index])
                    {
                        m_stepSize[index] *= GRADIENT_DESCENT_BETA;
                        ++countReduce;
                    }
                    m_preObjValue[index] = curObj;
                }
            }
        LOG(INFO) << countJump << " stepsize jumpped; " << countReduce <<" stepsize reduced.";
    }
}

void OPC::updateMask_cos()
{
    int index;
    for (int y = LITHOSIM_OFFSET; y < MASK_TILE_END_Y; ++y)
        for (int x = LITHOSIM_OFFSET; x < MASK_TILE_END_X; ++x)
        {
            index = getIndex(x, y);
            m_mask[index] = 0.5 + cos(m_params[index]) / 2.0;
        }
}

void OPC::updateMask_sig()
{
    int index;
    for (int y = LITHOSIM_OFFSET; y < MASK_TILE_END_Y; ++y)
        for (int x = LITHOSIM_OFFSET; x < MASK_TILE_END_X; ++x)
        {
            index = getIndex(x, y);
            m_mask[index] = 1.0 / (1.0 + exp(-MASKRELAX_SIGMOID_STEEPNESS * m_params[index]));
            //simply apply binary mask???
            //      if (m_params[index] > 0)
            //        m_mask[index] = 0.99;
            //      else
            //        m_mask[index] = 0.01;
        }
}

void OPC::determineEPEWeight(int numIteration)
{
    int index;
    if (numIteration == 1)
    {
        m_epeChecker.setEPESafeRegion(m_epeWeight, 10);
        //m_epeChecker.setEPESafeRegion(m_epeWeight, 15);
        for (int y = 0; y < OPC_TILE_Y; ++y)
            for (int x = 0; x < OPC_TILE_X; ++x)
            {
                index = getIndex(x, y);
                if (m_epeWeight[index] < ZERO_ERROR) //EPE region
                    m_epeWeight[index] = WEIGHT_EPE_REGION;
                //        else  //safe region
                //          m_epeWeight[index] = 2;
            }
    }
    else if (numIteration == 5)
    {
        for (int y = 0; y < OPC_TILE_Y; ++y)
            for (int x = 0; x < OPC_TILE_X; ++x)
            {
                index = getIndex(x, y);
                m_epeWeight[index] = 1; //reduce weight
            }
    }
}

void OPC::determineEPEWeightBndy(int numIteration)
{
    int index;
    if (numIteration == 1)
    {
        m_epeChecker.setEPEBoundary(m_epeWeight, 15);
        for (int y = 0; y < OPC_TILE_Y; ++y)
            for (int x = 0; x < OPC_TILE_X; ++x)
            {
                index = getIndex(x, y);
                if (m_epeWeight[index] > ZERO_ERROR) //EPE region
                    m_epeWeight[index] = WEIGHT_EPE_REGION;
                else //safe region
                    m_epeWeight[index] = 1;
            }
    }
    else if (numIteration == 5)
    {
        for (int y = 0; y < OPC_TILE_Y; ++y)
            for (int x = 0; x < OPC_TILE_X; ++x)
            {
                index = getIndex(x, y);
                m_epeWeight[index] = 1; //reduce weight
            }
    }
}

//keepout region should be leave empty
void OPC::findKeepOutRegion()
{
    int index;
    int up, down, left, right;

    memset(m_keepOutRegion, 0, sizeof(float) * OPC_TILE_SIZE); //all bits 1
    int keepOutStart = m_minWidth * (1.0 - OPC_KEEPOUT_REGION_RATIO) / 2.0;
    int keepOutEnd = keepOutStart + m_minWidth * OPC_KEEPOUT_REGION_RATIO;
    LOG(INFO) << "keepout region: " << keepOutStart - keepOutEnd;

    int *visited = new int[OPC_TILE_SIZE];
    memset(visited, 0, sizeof(int) * OPC_TILE_SIZE);
    for (int y = LITHOSIM_OFFSET; y < MASK_TILE_END_Y; ++y)
        for (int x = LITHOSIM_OFFSET; x < MASK_TILE_END_X; ++x)
        {
            index = getIndex(x, y);
            if (isPixelOn(index)) //occupied by a feature
            {
                up = index + OPC_TILE_X;
                down = index - OPC_TILE_X;
                right = index + 1;
                left = index - 1;
                if (!isPixelOn(up)) //pixel is on feature boundary
                    for (int i = keepOutStart; i < keepOutEnd; ++i)
                        ++visited[index + i * OPC_TILE_X];
                //m_keepOutRegion[index + i * OPC_TILE_X] = 1;
                if (!isPixelOn(down)) //pixel is on feature boundary
                    for (int i = keepOutStart; i < keepOutEnd; ++i)
                        ++visited[index - i * OPC_TILE_X];
                //m_keepOutRegion[index - i * OPC_TILE_X] = 1;
                if (!isPixelOn(right)) //pixel is on feature boundary
                    for (int i = keepOutStart; i < keepOutEnd; ++i)
                        ++visited[index + i];
                //m_keepOutRegion[index + i] = 1;
                if (!isPixelOn(left)) //pixel is on feature boundary
                    for (int i = keepOutStart; i < keepOutEnd; ++i)
                        ++visited[index - i];
                //m_keepOutRegion[index - i] = 1;
            }
        }

    for (int i = 0; i < OPC_TILE_SIZE; ++i)
        if (visited[i] > 1) //intersect by more than two features
            m_keepOutRegion[i] = 1;
    delete visited;
    //m_lithosim.writeImagePng("fig.png", m_keepOutRegion, 1);
}

//clear keepout region to avoid bridging between features
void OPC::keepoutBridge_sig()
{
    for (int i = 0; i < OPC_TILE_SIZE; ++i)
        if (m_keepOutRegion[i])
            m_params[i] = -1; //force mask to be 0
}

void OPC::keepBestResult(float curObj)
{
    if (curObj < m_minObjValue)
    {
        m_minObjValue = curObj;
        memcpy(m_bestMask, m_mask, sizeof(float) * OPC_TILE_SIZE);
    }
}

void OPC::restoreBestResult(float curObj)
{
    if (curObj > m_minObjValue)
    {
        LOG(INFO) << "Restore previous best solution";
        memcpy(m_mask, m_bestMask, sizeof(float) * OPC_TILE_SIZE);
    }
}

void OPC::updateAvgIntensity(const float *intensity, int numIteration)
{
    int index;
    for (int y = LITHOSIM_OFFSET; y < MASK_TILE_END_Y; ++y)
        for (int x = LITHOSIM_OFFSET; x < MASK_TILE_END_X; ++x)
        {
            index = getIndex(x, y);
            m_avgIntensity[index] =
                (m_avgIntensity[index] * (numIteration - 1) + intensity[index]) / numIteration;
        }
}

void OPC::retarget(int numIteration)
{
    int index;
    float delta;
    for (int y = LITHOSIM_OFFSET; y < MASK_TILE_END_Y; ++y)
        for (int x = LITHOSIM_OFFSET; x < MASK_TILE_END_X; ++x)
        {
            index = getIndex(x, y);
            //if (isPixelOn(m_targetImage[index]) != isPixelIntensityOn(m_avgIntensity[index]))
            //on->off
            if (isPixelOn(m_targetImage[index]) && !isPixelIntensityOn(m_avgIntensity[index]))
                delta = fabs(m_avgIntensity[index] - TARGET_INTENSITY);
            else
                delta = 0;
            //flip pixel if avg intensity if far from target
            if (delta > OPC_RETARGET_INTENSITY_THRESHOLD)
                m_targetImage[index] = 1 - m_targetImage[index];
        }

#ifdef _DEBUG
    std::stringstream ss;
    ss << numIteration;
    std::string fn1 = "retarget" + ss.str() + ".png";
    m_lithosim.writeImagePng((char *)fn1.c_str(), m_targetImage, 1);
#endif
}

//do value - matrix
void OPC::subValue2Matrix(float value, float *source, float *target)
{
    for (int i = 0; i < OPC_TILE_SIZE; ++i)
        target[i] = value - source[i];
}

//do matrix - value
void OPC::subMatrix2Value(float *source, float value, float *target)
{
    for (int i = 0; i < OPC_TILE_SIZE; ++i)
        target[i] = source[i] - value;
}

//do matrix + matrix
void OPC::addMatrix2Matrix(float *matrix1, float *matrix2, float *target)
{
    for (int i = 0; i < OPC_TILE_SIZE; ++i)
        target[i] = matrix1[i] + matrix2[i];
}

//do matrix + matrix
void OPC::addMatrix2Matrix(kiss_fft_cpx *matrix1, kiss_fft_cpx *matrix2, kiss_fft_cpx *target)
{
    for (int i = 0; i < OPC_TILE_SIZE; ++i)
    {
        target[i].r = matrix1[i].r + matrix2[i].r;
        target[i].i = matrix1[i].i + matrix2[i].i;
    }
}

//do matrix - matrix
void OPC::subMatrix2Matrix(float *matrix1, float *matrix2, float *target)
{
    for (int i = 0; i < OPC_TILE_SIZE; ++i)
        target[i] = matrix1[i] - matrix2[i];
}

void OPC::subMatrix2Matrix(kiss_fft_cpx *matrix1, kiss_fft_cpx *matrix2, kiss_fft_cpx *target)
{
    for (int i = 0; i < OPC_TILE_SIZE; ++i)
    {
        target[i].r = matrix1[i].r - matrix2[i].r;
        target[i].i = matrix1[i].i - matrix2[i].i;
    }
}

//do value * matrix
void OPC::multValue2Matrix(float value, float *matrix, float *target)
{
    for (int i = 0; i < OPC_TILE_SIZE; ++i)
        target[i] = value * matrix[i];
}

//do element-by-element muliplication
void OPC::multMatrixE2E(float *matrix1, float *matrix2, float *target)
{
    for (int i = 0; i < OPC_TILE_SIZE; ++i)
        target[i] = matrix1[i] * matrix2[i];
}

//do element-by-element muliplication
void OPC::multMatrixE2E(float *matrix1, kiss_fft_cpx *matrix2, kiss_fft_cpx *target)
{
    for (int i = 0; i < OPC_TILE_SIZE; ++i)
    {
        target[i].r = matrix1[i] * matrix2[i].r;
        target[i].i = matrix1[i] * matrix2[i].i;
    }
}

void OPC::writeMatrix(std::string fileName, const kiss_fft_cpx *matrix)
{
    std::string realFile = fileName + "r";
    std::string imgFile = fileName + "i";
    std::fstream fileR(realFile.c_str(), std::fstream::out);
    std::fstream fileI(imgFile.c_str(), std::fstream::out);
    for (int j = 0; j < OPC_TILE_Y; ++j)
    {
        for (int i = 0; i < OPC_TILE_X; ++i)
        {
            fileR << matrix[j * OPC_TILE_X + i].r << " ";
            fileI << matrix[j * OPC_TILE_X + i].i << " ";
        }
        fileR << std::endl;
        fileI << std::endl;
    }
    fileR.close();
    fileI.close();
}

void OPC::dumpConvergence()
{
    std::fstream file("convergence", std::fstream::out);
    //first row: objValue
    for (int i = 0; i <= OPC_ITERATION_THRESHOLD; ++i)
        file << m_objConvergence[i] << " ";
    file << std::endl;
    //second row: pvband
    for (int i = 0; i <= OPC_ITERATION_THRESHOLD; ++i)
        file << m_pvband[i] << " ";
    file << std::endl;
    //third row: epe
    for (int i = 0; i <= OPC_ITERATION_THRESHOLD; ++i)
        file << m_epeConvergence[i] << " ";
    file << std::endl;
    //forth row: score
    for (int i = 0; i <= OPC_ITERATION_THRESHOLD; ++i)
        file << m_scoreConvergence[i] << " ";
    file.close();
}

void OPC::drawHammer(int x, int y, int value)
{
    for (int j = y - OPC_LENGTH_CORNER_RESHAPE; j < y + OPC_LENGTH_CORNER_RESHAPE; ++j)
        for (int i = x - OPC_LENGTH_CORNER_RESHAPE; i < x + OPC_LENGTH_CORNER_RESHAPE; ++i)
            m_mask[getIndex(i, j)] = value;
}

void OPC::drawSraf(int x, int y, int length, orient_t orient)
{
    int llx, lly;
    int forbidPos;
    if (orient == HORIZONTAL)
    {
        llx = x;
        if (isPixelOn(getIndex(x, y - OPC_SPACE_SRAF))) //add top
        {
            lly = y + OPC_SPACE_SRAF;
            forbidPos = lly + OPC_WIDTH_SRAF + OPC_SPACE_FORBID_SRAF;
        }
        else //add bottom
        {
            lly = std::max(LITHOSIM_OFFSET, y - OPC_SPACE_SRAF - OPC_WIDTH_SRAF);
            forbidPos = std::max(LITHOSIM_OFFSET, lly - OPC_SPACE_FORBID_SRAF);
        }

        //check if adding is valid
        bool isValid = true;
        int from = std::min(lly, forbidPos);
        int to = std::max(lly, forbidPos);
        for (int j = from; j <= to; ++j)
            for (int i = llx; i < llx + length && isValid; ++i)
                if (isPixelOn(getIndex(i, j)))
                    isValid = false;

        if (isValid)
        {
            for (int j = lly; j < lly + OPC_WIDTH_SRAF; ++j)
                for (int i = llx; i < llx + length; ++i)
                    m_mask[getIndex(i, j)] = 1;
        }
    }
    else //VERTICAL
    {
        lly = y;
        if (isPixelOn(getIndex(x - OPC_SPACE_SRAF, y))) //add right
        {
            llx = x + OPC_SPACE_SRAF;
            forbidPos = llx + OPC_WIDTH_SRAF + OPC_SPACE_FORBID_SRAF;
        }
        else //add left
        {
            llx = std::max(LITHOSIM_OFFSET, x - OPC_SPACE_SRAF - OPC_WIDTH_SRAF);
            forbidPos = std::max(LITHOSIM_OFFSET, llx - OPC_SPACE_FORBID_SRAF);
        }

        //check if adding is valid
        bool isValid = true;
        int from = std::min(llx, forbidPos);
        int to = std::max(llx, forbidPos);
        for (int j = lly; j < lly + length && isValid; ++j)
            for (int i = from; i <= to; ++i)
                if (isPixelOn(getIndex(i, j)))
                    isValid = false;

        if (isValid)
        {
            for (int j = lly; j < lly + length; ++j)
                for (int i = llx; i < llx + OPC_WIDTH_SRAF; ++i)
                    m_mask[getIndex(i, j)] = 1;
        }
    }
}

bool OPC::isConvex(Coordinate *pre, Coordinate *cur, Coordinate *next)
{
    return ((cur->x - pre->x) * (next->y - pre->y) -
            (cur->y - pre->y) * (next->x - pre->x)) > 0;
}

int OPC::calculatePatternArea()
{
    int area = 0;
    for (int i = 0; i < OPC_TILE_SIZE; ++i)
        if (isPixelOn(m_targetImage[i]))
            ++area;
    return area;
}

//note cpxTerm 0 1 2 have been used for input/output
void OPC::calcEpeGradient(kiss_fft_cpx *term1, kiss_fft_cpx *term2, kiss_fft_cpx *result)
{
    int index;
    float sigProduct;

    memset(result, 0, sizeof(kiss_fft_cpx) * OPC_TILE_SIZE);
    memset(m_cpxTerm[3], 0, sizeof(kiss_fft_cpx) * OPC_TILE_SIZE);

    //loop each sample
    for (std::vector<EpeSample>::iterator it = m_epeSamples->begin(); it != m_epeSamples->end();
         ++it)
    {
        //H * partial (t0 M*H')
        it->setPosition(term1, m_cpxTerm[3]);
        m_lithosim.convolveKernel(m_cpxTerm[3], m_cpxTerm[4],
                                  LithosimWrapper::LITHO_KERNEL_FOCUS, MAX_DOSE);
        it->resetPosition(m_cpxTerm[3]);

        //H' * partial (t0 M*H)
        it->setPosition(term2, m_cpxTerm[3]);
        m_lithosim.convolveKernel(m_cpxTerm[3], m_cpxTerm[5],
                                  LithosimWrapper::LITHO_KERNEL_FOCUS_CT, MAX_DOSE);
        it->resetPosition(m_cpxTerm[3]);

        //accumulate for each pixel
        sigProduct = it->getSigProduct();
        for (int y = LITHOSIM_OFFSET; y < MASK_TILE_END_Y; ++y)
            for (int x = LITHOSIM_OFFSET; x < MASK_TILE_END_X; ++x)
            {
                index = getIndex(x, y);

                result[index].r += sigProduct * (m_cpxTerm[4][index].r + m_cpxTerm[5][index].r);
                result[index].i += sigProduct * (m_cpxTerm[4][index].i + m_cpxTerm[5][index].i);
            }
    }
}

//calculate m_sigProduct
//given m_diffImage
int OPC::updateSampleEPE()
{
    int index;
    int sampleX, sampleY;
    int count = 0;
    int constraint = EPE_CONSTRAINT; // - 2;  //13
    for (std::vector<EpeSample>::iterator it = m_epeSamples->begin(); it != m_epeSamples->end(); ++it)
    {
        sampleX = it->getX();
        sampleY = it->getY();
        float epe = 0;
        if (it->getOrient() == EpeChecker::HORIZONTAL)
        {
            //check vertical direction
            //for (int y = sampleY - EPE_CONSTRAINT; y <= sampleY + EPE_CONSTRAINT; ++y)
            //count only 2 points
            //      for (int y = sampleY - EPE_CONSTRAINT; y <= sampleY + EPE_CONSTRAINT;
            //          y += 2 * EPE_CONSTRAINT)
            for (int y = sampleY - constraint; y <= sampleY + constraint;
                 y += 2 * constraint)
            {
                index = getIndex(sampleX, y);
                //epe += m_diffImage[index] * m_diffImage[index]; //(Z-Z0)^2
                //use binary value
                //if (isPixelOn(m_image[2][index]) ^ isPixelOn(m_targetImage[index]))
                if (isPixelIntensityOn(m_image[2][index]) ^ isPixelOn(m_targetImage[index]))
                {
                    epe += 1;
                }
            }
        }
        else //VERTICAL
        {
            //check horizontal direction
            //for (int x = sampleX - EPE_CONSTRAINT; x <= sampleX + EPE_CONSTRAINT; ++x)
            //count only 2 points
            //      for (int x = sampleX - EPE_CONSTRAINT; x <= sampleX + EPE_CONSTRAINT;
            //          x += 2 * EPE_CONSTRAINT)
            for (int x = sampleX - constraint; x <= sampleX + constraint;
                 x += 2 * constraint)
            {
                index = getIndex(x, sampleY);
                //epe += m_diffImage[index] * m_diffImage[index]; //(Z-Z0)^2
                //use binary value
                //if (isPixelOn(m_image[2][index]) ^ isPixelOn(m_targetImage[index]))
                if (isPixelIntensityOn(m_image[2][index]) ^ isPixelOn(m_targetImage[index]))
                {
                    epe += 1;
                }
            }
        }

        if (epe > EPE_CONSTRAINT_INTERNAL)
        {
            ++count;
            //dmesg("(%d, %d) ", sampleX-512, sampleY-512);
        }
        float epeSig = 1.0 / (1.0 + exp(-EPERELAX_SIGMOID_STEEPNESS *
                                        (epe - EPE_CONSTRAINT_INTERNAL)));
        it->setSigProduct(epeSig * (1 - epeSig));
        //dmesg("%f, %f, %f-", epe, epeSig, it->getSigProduct());
    }

    return count;
}

void OPC::dbgWriteBinaryImage(const char *fileName, float *image)
{
    std::fstream fs(fileName, std::fstream::out);

    for (int x = 0; x < OPC_TILE_X; ++x)
    {
        for (int y = 0; y < OPC_TILE_Y; ++y)
        {
            if (isPixelOn(image[getIndex(x, y)]))
                fs << "1 ";
            else
                fs << "0 ";
        }
        fs << std::endl;
    }
}

//sum up gradients of diff function within EPE range
//float
//OPC::getSampleGradient(EpeSample* sample, int targetX, int targetY)
//{
//  //load kernel!!!!!!!!!!
//  int xStart, xEnd;
//  int yStart, yEnd;
//  if (sample->getOrient() == EpeChecker::HORIZONTAL)
//  {
//    //scan vertically
//    xStart = sample->getX();
//    xEnd = xStart + 1;
//    yStart = sample->getY() - EPE_CONSTRAINT;
//    yEnd = sample->getY() + EPE_CONSTRAINT + 1;
//  }
//  else  //VERTICAL
//  {
//    //scan horizontally
//    xStart = sample->getX() - EPE_CONSTRAINT;
//    xEnd = sample->getX() + EPE_CONSTRAINT + 1;
//    yStart = sample->getY();
//    yEnd = yStart + 1;
//  }
//
//  float sum = 0;
//  for (int y = yStart; y < yEnd; ++y)
//    for (int x = xStart; x < xEnd; ++x)
//    {
//      sum += m_cpxTerm[0] * aa + m_cpxTerm[1] * bb;
//      LithosimWrapper::getKernel();
//    }
//}
