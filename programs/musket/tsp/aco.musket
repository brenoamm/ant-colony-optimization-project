#config PLATFORM GPU CUDA
#config PROCESSES 1
#config CORES 24
#config GPUS 1
#config MODE debug

const int ants = 256;
const int ncities = 256;
double bestroute = 99999999.9;
const int IROULETE = 32;
const double PHERINIT = 0.005;
const double EVAPORATION = 0.5;
const int ALPHA = 1;
const int BETA = 2;
const int TAUMAX = 2;
const int block_size = 64;
array<double,512,dist, yes> cities; // cities * 2
array<int,256,dist, yes> city; // cities * 2
array<int,256,dist, yes> antss; // pseudo array

array<double,65536,dist, yes> phero; // cities squared
array<double,65536,dist, yes> phero_new; // cities squared

array<double,65536,dist, yes> distance; // cities squared
array<int,512,dist, yes> best_sequence;

array<double,65536,dist, yes> d_delta_phero; // cities squared
array<double,256,dist, yes> d_routes_distance; //n_ants ? TODO ask change in model
array<double,65536,dist, yes> d_probabilities;//n_ants*n_cities

array<int,8192,dist, yes> d_iroulette; // IROULETTE * cities
array<int,65536,dist, yes> d_routes; //n_ants*n_cities

double writeIndex(int i, double y){
	return (double)i;
}

double calculate_distance(int i, double y){
	double returner = 0.0;
	int j = i / ncities;
	int currentcity = (int) i % ncities;
	if (j != currentcity) {
		double difference = cities[j*2] - cities[currentcity*2];
		if (difference < 0) {difference = difference * (-1) ;}
		double nextdifference = cities[(j*2) + 1] - cities[(currentcity*2) + 1];
		if (nextdifference < 0) { nextdifference = nextdifference * (-1) ;}
		double pow2 = 2.0;
		double first = mkt::pow(difference, pow2);
		double second = mkt::pow(nextdifference, pow2);
		returner = mkt::sqrt(first + second);
	}
	return returner;
}

int calculate_iroulette(int cityindex, int value){
	int c_index = cityindex;
	for(int i = 0 ; i< IROULETE ; i++) {
		double citydistance = 999999.9;
		double c_dist = 0.0;
		int cityy = -1;
		for(int j = 0 ;j<ncities;j++){
			bool check = true;
			for(int k = 0 ; k < i ; k++){
				if(d_iroulette[c_index * IROULETE + k] == j){check = false;	}
			}
			if(c_index != j){
				if (check == true) {
					c_dist = distance[(c_index * ncities) + j];
					if(c_dist < citydistance){
						citydistance = c_dist;
						cityy = j;
					}
				}
			}
		}
		d_iroulette[c_index * IROULETE + i] = cityy;
	}
	return value;
}

int route_kernel2(int Index, int value){
	int newroute = 0;
	int ant_index = Index/ants;
	double sum = 0.0;
	int next_city = -1;
	double ETA = 0.0;
	double TAU = 0.0;
	double random = 0.0;
	int initial_city = (int) random * ncities;
	d_routes[ant_index * ncities] = initial_city;
	if (ant_index < ants) {
		for (int i=0; i < ncities-1; i++) {
			int cityi = d_routes[ant_index * ncities + i];
			int count = 0;
			for (int c = 0; c < IROULETE; c++) {
		
				next_city =  d_iroulette[(cityi * IROULETE) + c];
				int visited = 0;
				for (int l=0; l <= i; l++) {
					if (d_routes[ant_index*ncities+l] == next_city){visited = 1;}
				}
				if (!visited){
					int indexpath = (cityi * ncities) + next_city;
					double firstnumber = 1 / distance[indexpath];
					ETA = (double) mkt::pow(firstnumber, BETA);
					TAU = (double) mkt::pow(phero[indexpath], ALPHA);
					sum += ETA * TAU;
				}	
			}
			for (int c = 0; c < IROULETE; c++) {
				next_city = d_iroulette[(cityi * IROULETE) + c];
				int visited = 0;
				for (int l=0; l <= i; l++) {
					if (d_routes[ant_index*ncities+l] == next_city) {visited = 1;}
				}
				if (visited) {
					d_probabilities[ant_index*ncities+c] = 0.0;
				} else {
					double dista = (double)distance[cityi*ncities+next_city];
					double ETAij = (double) mkt::pow(1 / dista , BETA);
					double TAUij = (double) mkt::pow(phero[(cityi * ncities) + next_city], ALPHA);			
					d_probabilities[ant_index*ncities+c] = (ETAij * TAUij) / sum;
					count = count + 1;
				}
			}
		
			if (0 == count) {
				int breaknumber = 0;
				for (int nc = 0; nc < (ncities); nc++) {
                    int visited = 0;
                    for (int l = 0; l <= i; l++) {
                        if (d_routes[(ant_index * ncities) + l] == nc) { visited = 1;}
                    }
                    if (!(visited)) {
                        breaknumber = (nc);
                        nc = ncities;
                    }
                }
				newroute = breaknumber;
			} else {
				random = mkt::rand(0.0, (double)ncities);
				int ii = -1;
				double summ = d_probabilities[ant_index*ncities];
			
				for(int check = 1; check > 0; check++){
					if (summ >= random){
						check = -2;
					} else {
						i = i+1;
						summ += d_probabilities[ant_index*ncities+ii];
					}
				}
				int chosen_city = ii;
				newroute = d_iroulette[cityi*IROULETE+chosen_city];
			}
			d_routes[(ant_index * ncities) + (i + 1)] = newroute;
			sum = 0.0;
		}
	
	}
	return value;
}

int update_best_sequence_kernel(int Index, int value) {
	int Q = 11340;
	double RO = 0.5;
	int k = Index;
	double rlength = 0.0;
	double sum = 0.0;
	if (Index <= ants) {
		for (int j=0; j<ncities-1; j++) {
	
			int cityi_infor = d_routes[k*ncities+j];
			int cityj_infor = d_routes[k*ncities+j+1];
	
			sum += distance[cityi_infor*ncities + cityj_infor];
		}
	
		int cityi_old = d_routes[k*ncities+ncities-1];
		int cityj_old = d_routes[k*ncities];
	
		sum += distance[cityi_old*ncities + cityj_old];
		
		d_routes_distance[k] = sum ;
		for (int r=0; r < ncities-1; r++) {
	
			int cityi = d_routes[k * ncities + r];
			int cityj = d_routes[k * ncities + r + 1];
			double delta = d_delta_phero[cityi * ncities + cityj];
			d_delta_phero[cityi* ncities + cityj] = delta + (Q / sum);
			d_delta_phero[cityj* ncities + cityi] = delta + (Q / sum);
		}		
	}
	return value;
}

int update_pheromones1(double bestRoute, int Index, int value) {
	int k = Index;		
	if(d_routes_distance[k] == bestRoute){
		bestRoute = d_routes_distance[k];
		for (int count=0; count < ncities; count++) {
			best_sequence[count] = d_routes[k * ncities+count];
		}
	}
	return value;
}

int update_pheromones2(int Index, int value) {
	int Q = 11340;
	double RO = 0.5;
	int i = Index;
	for (int j=0; j<ncities; j++) {
		double new_phero =  (1 - RO) * phero[(i * ncities) + j] + d_delta_phero[(i * ncities) + j];
    	if(new_phero > 2.0){new_phero = 2.0;}
    	if(new_phero < 0.1){new_phero = 0.1;}
        phero[(i * ncities) + j] = new_phero;
        phero[(j * ncities) + i] = new_phero;
        d_delta_phero[(i * ncities) + j] = 0.0;
        d_delta_phero[(j * ncities) + i] = 0.0;
	}
	return value;
}

main{
	mkt::roi_start();
	cities.mapIndexInPlace(writeIndex());
	distance.mapIndexInPlace(calculate_distance());
	city.mapIndexInPlace(calculate_iroulette());
	int iterations = 2;
	for (int i = 0; i < iterations; i++){
		antss.mapIndexInPlace(route_kernel2());
		antss.mapIndexInPlace(update_best_sequence_kernel());
		bestroute = d_routes_distance.reduce(min);
		antss.mapIndexInPlace(update_pheromones1(bestroute));
		city.mapIndexInPlace(update_pheromones2());
	}
	mkt::roi_end();
}
