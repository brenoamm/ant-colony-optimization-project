// ===  FUNCTION  ======================================================================
//         Name:  run
//  Description:  Runs aco with the parameter set regarding the problem, number of ants
//				  number of iterations, number threads. It runs several times.
// =====================================================================================
double run(const std::string& problem, int nants, int iterations);

void readMap(double* coord, double* phero, const std::string& problem);

void calcDist(double* dist, double* coord);

void route(double* phero, double* dist, int sequence[]);

void pheroEvap(double* phero);

double pheroDeposit(int* sequence, double* dist, double* phero);

bool vizited (int* route, int cityc, int count);

void report_num_threads(int level);
