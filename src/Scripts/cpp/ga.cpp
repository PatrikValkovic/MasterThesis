#include <iostream>
#include <fstream>
#include <thread>
#include <cstdlib>
#include <string>
#include <vector>

using namespace std;

int main(int argc, char** args) {
    // help
    for(int i=0;i<argc;i++){
        if(strcmp(args[i], "--help") == 0 || argc != 7){
            cout << "Genetic algorithm solving 3SAT problem" << endl;
            cout << "usage: " << args[0] << " literals clauses repeat iterations popsize[,popsize[,...]] outputfile" << endl;
            return 0;
        }
    }
    // parse args
    int literals = atoi(args[1]);
    int clauses = atoi(args[2]);
    int repeat = atoi(args[3]);
    int iterations = atoi(args[4]);
    string outputfile = args[6];
    vector<int> popsize;
    int current_index = 0;
    int param_len = strlen(args[5]);
    while(current_index < param_len){
        int next_position = find(args[5]+current_index, args[5]+param_len, ',');
        args[5][next_position] = '\0';
        int pop = atoi(args[5]+current_index);
        popsize.push_back(pop);
        current_index = next_position + 1;
    }
    cout << literals << endl;
    cout << clauses << endl;
    cout << repeat << endl;
    cout << iterations << endl;
    for(int p: popsize)
        cout << p << ";";
    cout << endl;
    cout << literals << endl;
    cout << outputfile << endl;
}
