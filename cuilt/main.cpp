#include <iostream>
#include <sys/time.h>
#include <glog/logging.h>
#include <gflags/gflags.h>
#include "shapes/design.h"
#include "ilt/opc.h"
#include "utils/debug.h"
#include "utils/exception.h"

void printWelcome();
void printUsage();

DEFINE_string(input, "",
"Required; the target design pattern");
DEFINE_string(output, "",
"Required; specify the output optimized mask");
DEFINE_bool(gpu, false,
"Optional; with GPU-based lithography simulation");
DEFINE_string(method, "mosaic",
"Optional; select MOSAIC method or Level-set method for optimization, (MOSAIC by default)");
DEFINE_int32(iterations, 20,
"Optional; total number of iterations to run");

/* Please note that this is a Global variable*/
/* It is declared here and used in pvbandsim.cpp*/
/* to determine whether to use gpu */
extern int USE_GPU;

int main(int argc, char *argv[])
{
    FLAGS_alsologtostderr = 1;
    google::InitGoogleLogging(argv[0]);
    google::InstallFailureSignalHandler();
    gflags::ParseCommandLineFlags(&argc, &argv, true);
    printWelcome();
    try
    {
        if(FLAGS_gpu){
            USE_GPU = 1;
            LOG(INFO)<< "Lithography with GPU";
        }else{
            USE_GPU = 0;
        }

        struct timeval startTime, endTime;
        gettimeofday(&startTime, NULL);
        Design design(FLAGS_input.c_str());
        OPC opc(design);
        opc.run();
        design.writeGlp(FLAGS_output.c_str());

#ifdef _DEBUG
        dmesg("==============System statistics==============\n");
        gettimeofday(&endTime, NULL);
        float runtime = (endTime.tv_sec - startTime.tv_sec) +
                        (float)(endTime.tv_usec - startTime.tv_usec) / 1e6;
        int pvband = opc.getPvband();
        int numEpe = opc.getNumEpe();
        dmesg("Runtime: %f sec\n", runtime);
        dmesg("Pvband: %d nm^2\n", pvband);
        dmesg("#EPE violations of nominal image: %d\n", numEpe);
        dmesg("Score (single CPU): %f\n",
              runtime + 5000 * numEpe);
        dmesg("=============================================\n");
#endif
    }
    catch (int error)
    {
        switch (error)
        {
        case EXCEPTION_NOT_ENOUGH_INPUT:
            printUsage();
            break;
        case EXCEPTION_OPEN_DESIGN_ERROR:
            LOG(FATAL) << "[FATAL]: open design failed";
            break;
        default:
            LOG(ERROR) << "[ERROR]: throw an undefined exception here...";
            break;
        }
    }
    catch (BaseException &e)
    {
        e.printLog();
    }

    return 0;
}

void printWelcome()
{
    std::cout << std::endl
        << "******************   CUILT   ********************" << std::endl
        << "*************************************************" << std::endl
        << std::endl;
}

void printUsage()
{
    std::cout << std::endl
        << "Usage:" << std::endl
        << "main -input <in_layout_file> -output <out_layout_file>" << std::endl
        << std::endl;
}
