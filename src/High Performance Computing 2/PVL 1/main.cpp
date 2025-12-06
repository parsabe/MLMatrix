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
        std::cout << "error: " << abs_diff << "\n";
        std::cout << "\n";
    }
}

int main() {
    const int intervals = 200;       
    const int cores = 5;
    int thread_choices[cores] = {1, 2, 4, 8, 12};

    std::cout.precision(17);
    std::cout << "Setting up the methods:\n";

    run_method("Trapezoid", true, intervals, thread_choices, cores);

    std::cout << "----------------------------------------\n";

    run_method("Simpson", false, intervals, thread_choices, cores);

    return 0;
}
