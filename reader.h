#include <fstream>
#include <vector>

typedef struct point_charge {
	double x;
	double y;
	double charge;
} point_charge_t;

class input_reader {
private:
	std::ifstream input_stream;
public:
	input_reader(std::string file_name);
	std::vector<point_charge_t> read_contents();
};
