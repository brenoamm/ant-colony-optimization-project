/*
 * timer.h
 *
 *      Author: Steffen Ernsting <s.ernsting@uni-muenster.de>
 *
 * -------------------------------------------------------------------------------
 *
 * The MIT License
 *
 * Copyright 2014 Steffen Ernsting <s.ernsting@uni-muenster.de>,
 *                Herbert Kuchen <kuchen@uni-muenster.de.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 */

#pragma once
#ifndef ACO_LOW_LEVEL_TIMER_H
#define ACO_LOW_LEVEL_TIMER_H

#endif //ACO_LOW_LEVEL_TIMER_H

#include <mpi.h>
#include <sstream>

namespace msl {

/**
 * \brief Class Timer for timing purposes.
 */
    class Timer {
    public:
        /**
         * \brief Default constructor.
         */
        Timer() : start(MPI_Wtime()), split(MPI_Wtime()), end(0.0), splits(0) {}

        /**
         * Creates a timer with name \em n.
         *
         * @param n The name of the timer.
         */
        Timer(const std::string &n)
                : start(MPI_Wtime()), split(MPI_Wtime()), end(0.0), splits(0), name(n) {}

        /**
         * \brief Stops the timer.
         *
         * @return Elapsed time since start.
         */
        double stop() {
            MPI_Barrier(MPI_COMM_WORLD);
            end = MPI_Wtime();

            return end - start;
        }

        /**
         * \brief Sets a split time.
         *
         * @return Elapsed time since last split.
         */
        double splitTime() {
            MPI_Barrier(MPI_COMM_WORLD);
            double result = MPI_Wtime() - split;
            split = MPI_Wtime();
            splits++;
            return result;
        }

        /**
         * \brief Returns the total elapsed time.
         *
         * @return The total elapsed time.
         */
        double totalTime() {
            if (end == 0.0) {
                return stop();
            } else {
                return end - start;
            }
        }

        /**
         * \brief Returns the number of splits.
         *
         * @return The number of splits.
         */
        int getNumSplits() { return splits; }

    private:
        double start, split, end;
        int splits;
        std::string name;
    };

} // namespace msl
