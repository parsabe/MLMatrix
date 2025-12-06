

<h1>C++ Code</h1>
<pre>
#include <iostream>
#include <cmath>
#include <omp.h>

double g(double x_val) {
    return 1.0 / (1.0 + std::pow(x_val, 2));
}

double trapzoid(int intervals, int thread_num) {
    double xstart = 0.0;
    double xend   = 1.0;
    double step    = (xend - xstart) / intervals;

    double accum = 0.5 * (g(xstart) + g(xend));

#pragma omp parallel for reduction(+:accum) num_threads(thread_num)
    for (int idx = 1; idx < intervals; ++idx) {
        double x_local = xstart + idx * step;
        accum += g(x_local);
    }

    return accum * step;
}

double simpson(int intervals, int thread_num) {
    double xstart = 0.0;
    double xend   = 1.0;

    if (intervals % 2 != 0) {
        return std::nan("");
    }

    double step = (xend - xstart) / intervals;
    double total = g(xstart) + g(xend);

#pragma omp parallel for reduction(+:total) num_threads(thread_num)
    for (int idx = 1; idx < intervals; ++idx) {
        double x_local = xstart + idx * step;
        if (idx % 2 == 0) {
            total += 2.0 * g(x_local);
        } else {
            total += 4.0 * g(x_local);
        }
    }

    return total * step / 3.0;
}

void run_method(const std::string &label,
                bool use_trap,
                int intervals,
                const int *thread_list,
                int thread_count) {
    double pi_exact = std::acos(-1.0);
    double base_runtime = 0.0;

    std::cout << label << "\n";
    std::cout << "Intervals: " << intervals << "\n\n";

    for (int i = 0; i < thread_count; ++i) {
        int t_current = thread_list[i];

        double t_begin = omp_get_wtime();

        double integral_val = 0.0;
        if (use_trap) {
            integral_val = trapzoid(intervals, t_current);
        } else {
            integral_val = simpson(intervals, t_current);
        }

        double t_end = omp_get_wtime();
        double elapsed = t_end - t_begin;

        if (i == 0) {
            base_runtime = elapsed;
        }

        double pi_approx = 4.0 * integral_val;
        double abs_diff = std::fabs(pi_approx - pi_exact);
        double speed = base_runtime / elapsed;

        std::cout << "thread: " << t_current << "\n";
        std::cout << "time: " << elapsed << "\n";
        std::cout << "speed: " << speed << "\n";
        std::cout << "pi: " << pi_approx << "\n";
        std::cout << "error: " << abs_diff << "\n\n";
    }
}

int main() {
    const int intervals = 200;
    const int cores = 5;
    int thread_choices[cores] = {1, 2, 4, 8, 12};

    std::cout.precision(17);
    std::cout << "Setting up the methods:\n\n";

    run_method("Trapezoid", true, intervals, thread_choices, cores);

    std::cout << "----------------------------------------\n";

    run_method("Simpson", false, intervals, thread_choices, cores);

    return 0;
}
</pre>


<h1>PBS Job Script</h1>
<pre>
#!/bin/bash
#PBS -N pvl1-python
#PBS -q teachingq
#PBS -l select=1:ncpus=1:mpiprocs=1
#PBS -l walltime=00:05:00
#PBS -o log.out
#PBS -e log.err

module load openmpi/gcc/11.4.0/4.1.6
./main.bin
</pre>


<h1>Program Output</h1>
<pre>
Setting up the methods:
Trapezoid
Intervals: 200

thread: 1
time: 6.9630332291126251e-06
speed: 1
pi: 3.1415884869231299
error: 4.166666663252272e-06

thread: 2
time: 2.8101960197091103e-05
speed: 0.24777749239831975
pi: 3.1415884869231263
error: 4.1666666668049857e-06

thread: 4
time: 6.2366947531700134e-05
speed: 0.11164620852372846
pi: 3.1415884869231268
error: 4.1666666663608964e-06

thread: 8
time: 0.00010557984933257103
speed: 0.065950399371942969
pi: 3.1415884869231263
error: 4.1666666668049857e-06

thread: 12
time: 0.00011448003351688385
speed: 0.060823123606840113
pi: 3.1415884869231263
error: 4.1666666668049857e-06

----------------------------------------
Simpson
Intervals: 200

thread: 1
time: 1.4640390872955322e-06
speed: 1
pi: 3.1415926535897949
error: 1.7763568394002505e-15

thread: 2
time: 5.4629985243082047e-05
speed: 0.02679918511383687
pi: 3.1415926535897936
error: 4.4408920985006262e-16

thread: 4
time: 4.365481436252594e-05
speed: 0.033536715449929595
pi: 3.1415926535897931
error: 0

thread: 8
time: 7.7238772064447403e-05
speed: 0.018954717276887183
pi: 3.1415926535897927
error: 4.4408920985006262e-16

thread: 12
time: 0.00010359100997447968
speed: 0.014132877820731816
pi: 3.1415926535897927
error: 4.4408920985006262e-16
</pre>

<p>
The absolute errors clearly show that the Trapezoid method delivers an accuracy on the order of 10⁻⁶, while Simpson’s method achieves near machine-precision errors around 10⁻¹⁵. This confirms that Simpson’s rule is significantly more accurate for the same number of intervals, and the error remains stable across all thread counts since parallelism affects runtime, not numerical precision.
</p>


</body>
</html>

