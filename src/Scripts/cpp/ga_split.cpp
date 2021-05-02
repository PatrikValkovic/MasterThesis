#include <iostream>
#include <fstream>
#include <thread>
#include <cstdlib>
#include <string>
#include <vector>
#include <cstring>
#include <mutex>
#include <queue>
#include <condition_variable>
#include <random>
#include <ctime>
#include <chrono>

using namespace std;
using ulock = unique_lock<mutex>;

struct Problem {
    int literals;
    int clauses;
    int* literals_in_clause;
    int** formula;
    bool* solution;
};

struct EvalStruct {
    Problem* problem;
    bool* individual;
    int* unsatisfied;
};

class BlockingQueue {
private:
    mutex _m;
    queue<EvalStruct> _queue;
    int _waiting_threads = 0;
    condition_variable _queue_push;
public:
    void push(EvalStruct &str){
        {
            ulock guard(_m);
            _queue.push(str);
        }
        _queue_push.notify_one();
    }

    EvalStruct pull(){
        ulock guard(_m);
        _queue_push.wait(guard, [this]{return !_queue.empty();});
        EvalStruct x = _queue.front();
        _queue.pop();
        return x;
    }
};

class MyCounter {
private:
    int _counter = 0;
    mutex _m;
    condition_variable _var;
public:
    int fire_at;
    MyCounter(int fire_at): fire_at(fire_at){}

    void reset(){
        ulock l(_m);
        _counter = 0;
    }

    void wait(){
        ulock l(_m);
        _var.wait(l, [this]{return _counter == fire_at;});
    }

    void increment(){
        bool notify;
        {
            ulock l(_m);
            _counter++;
            notify = _counter == fire_at;
        }
        if(notify)
            _var.notify_all();
    }
};

void eval_thread(BlockingQueue* queue, MyCounter* executed){
    while(true){
        EvalStruct eval = queue->pull();
        Problem* problem = eval.problem;
        int unsatisfied = 0;
        for(int c=0;c<problem->clauses;c++){
            bool satisfied = false;
            for(int l=0;l<problem->literals_in_clause[c];l++){
                int lit = problem->formula[c][l];
                int lit_index = abs(lit)-1;
                bool cur_satisfied = eval.individual[lit_index];
                if(lit < 0){
                    cur_satisfied = !cur_satisfied;
                }
                satisfied = satisfied || cur_satisfied;
            }
            if(!satisfied)
                unsatisfied++;
        }
        *eval.unsatisfied = unsatisfied;
        executed->increment();
    }
}

Problem create_problem(int literals, int clauses, int literals_in_clause){
    bool* solution = new bool[literals];
    for(int i=0;i<literals;i++)
        solution[i]= (rand() % 2) == 0;
    int* liters_in_c =new int[clauses];
    int** formula = new int*[clauses];
    for(int i=0;i<clauses;i++){
        liters_in_c[i] = literals_in_clause;
        formula[i]=new int[literals_in_clause];
        for(int j=0;j<liters_in_c[i];j++){
            bool is_already_there = true;
            int lit = 0;
            while(is_already_there){
                lit = rand() % literals + 1;
                is_already_there = false;
                for(int k=0;k<j;k++){
                    is_already_there = is_already_there || formula[i][k] == lit;
                    is_already_there = is_already_there || formula[i][k] == -lit;
                }
                lit -= 1;
            };
            bool is_pos = solution[lit];
            lit++;
            if(!is_pos)
                lit *= -1;
            formula[i][j]=lit;
        }
    }
    Problem p;
    p.literals = literals;
    p.clauses = clauses;
    p.literals_in_clause = liters_in_c;
    p.formula = formula;
    p.solution = solution;
    return p;
}

void delete_problem(Problem problem){
    delete [] problem.literals_in_clause;
    for(int i=0;i<problem.clauses;i++)
        delete [] problem.formula[i];
    delete [] problem.formula;
    delete [] problem.solution;
}


bool early_terminate(int iter, int min_iters, int max_time, chrono::time_point<chrono::high_resolution_clock> &start){
    auto now = chrono::high_resolution_clock::now();
    chrono::duration<double> running_for = now-start;
    return iter > min_iters && running_for.count() > max_time;
}

void evaluate(MyCounter& counter, int psize, Problem &p, int literals, bool* pop, int* fitness, BlockingQueue &queue){
    counter.reset();
    for(int i=0;i<psize;i++){
        EvalStruct e;
        e.problem = &p;
        e.individual = pop+i*literals;
        e.unsatisfied = fitness+i;
        queue.push(e);
    }
    counter.wait();
}

void selection(int psize, int* fitness, bool* &pop, bool* &second_pop, int literals){
    for(int i=0;i<psize;i++){
        int first= rand() % psize;
        int second = rand() % psize;
        int better = fitness[first] < fitness[second] ? first : second;
        memcpy(second_pop+i*literals, pop+better*literals, literals * sizeof(bool));
    }
    swap(second_pop, pop);
}

void crossover(int psize, int literals, bool* crossover_tmp, bool* pop){
    int to_crossover_pop = 0.4 * psize;
    for(int i=0;i<to_crossover_pop;i++){
        int p1 = rand() % psize;
        int p2 = p1;
        while(p2 == p1){
            p2 = rand() % psize;
        }
        int split = rand() % (literals - 1) + 1;
        memcpy(crossover_tmp, pop+p1*literals+split,literals-split);
        memcpy(pop+p1*literals+split, pop+p2*literals+split, literals-split);
        memcpy(pop+p2*literals+split, crossover_tmp, literals-split);
    }
}

void mutation(int psize, int literals, bool* pop){
    int to_mutate_pop = 0.6 * psize;
    for(int i=0;i<to_mutate_pop;i++){
        int to_mutate = rand() % psize;
        for(int j=0;j<literals;j++){
            if(rand() % 1000 == 0){
                pop[to_mutate * literals + j] = !pop[to_mutate * literals + j];
            }
        }
    }
}

void alg(vector<int> popsize,
         int literals,
         int clauses,
         int repeat,
         int min_iters,
         int max_time,
         MyCounter& counter,
         BlockingQueue &queue,
         ofstream& outfile,
         int num_threads,
         int iterations
){
    for(int psize : popsize){
        counter.fire_at = psize;
        int iter = 0;
        for(int rep=0;rep < repeat;rep++){
            cout << "Running SAT for population of size " << psize << " for the " << rep << " time" << endl;
            Problem p = create_problem(literals, clauses, 3);
            auto start = chrono::high_resolution_clock::now();
            // allocation
            bool* pop = new bool[psize * literals];
            bool* second_pop = new bool[psize * literals];
            bool* crossover_tmp = new bool[literals];
            for(int i=0;i<psize*literals;i++)
                pop[i] = (rand() % 2) == 0;
            int* fitness = new int[psize];
            for(iter=0;iter<iterations;iter++){
                // early termination
                if(early_terminate(iter, min_iters, max_time, start))
                    break;
                // evaluate
                evaluate(counter, psize, p, literals,pop,fitness,queue);
                /*int mi = clauses;
                double me = 0;
                for(int i=0;i<psize;i++){
                    mi = min(mi, fitness[i]);
                    me += fitness[i];
                }
                me = me / psize;
                cout << "MEAN:" << me << endl << "MIN :" << mi << endl;*/
                // selection
                selection(psize, fitness, pop, second_pop, literals);
                // crossover
                crossover(psize, literals, crossover_tmp, pop);
                // mutation
                mutation(psize, literals, pop);
            }
            delete [] fitness;
            delete [] pop;
            delete [] second_pop;
            delete [] crossover_tmp;
            auto end = chrono::high_resolution_clock::now();
            chrono::duration<double> running_time = end-start;
            // report
            // popsize;literals;clauses;iterations;threads;executed_iterations;time
            outfile << psize << ';' << literals << ';' << clauses << ';' << iterations << ';';
            outfile << num_threads << ';' << iter << ';';
            outfile << running_time.count() << endl;
            delete_problem(p);
        }
    }
}


int main(int argc, char** args) {
    // help
    for(int i=0;i<argc;i++){
        if(strcmp(args[i], "--help") == 0 || argc != 10){
            cout << "Genetic algorithm solving 3SAT problem" << endl;
            cout << "usage: " << args[0] << " literals clauses repeat iterations popsize[,popsize[,...]] outputfile threads min_iteration max_time" << endl;
            return 0;
        }
    }
    // parse args
    int literals = atoi(args[1]);
    int clauses = atoi(args[2]);
    int repeat = atoi(args[3]);
    int iterations = atoi(args[4]);
    string outputfile = args[6];
    int num_threads = atoi(args[7]);
    int min_iters = atoi(args[8]);
    int max_time = atoi(args[9]);
    vector<int> popsize;
    char *current_index = args[5], *last_index = args[5]+strlen(args[5]);
    while(current_index < last_index){
        char* next_position = strchr(current_index, ',');
        if(next_position == nullptr)
            next_position = last_index;
        *next_position = '\0';
        int pop = atoi(current_index);
        popsize.push_back(pop);
        current_index = next_position + 1;
    }
    cout << "Arguments parsed" << endl;
    // prepare file and seed
    srand(time(nullptr));
    ofstream outfile(outputfile, std::ios::out | std::ios::app);
    cout << "File opened" << endl;
    // prepare threads and queue
    MyCounter counter(0);
    BlockingQueue queue;
    vector<thread> threads;
    for(int i=0;i<num_threads;i++){
        threads.emplace_back(eval_thread, &queue, &counter);
        threads[i].detach();
    }
    // run the algorithm
    alg(popsize, literals, clauses, repeat, min_iters, max_time, counter, queue,outfile,num_threads, iterations);    // clean
    exit(0);
}
