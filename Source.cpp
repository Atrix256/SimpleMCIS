#include <stdio.h>
#include <random>
#include <stdlib.h>

std::random_device g_rd;
std::mt19937 g_mt(g_rd());

static const float c_pi = 3.14159265359f;

// y = sin(x)^2
struct SinX_Squared
{
    static const char* Name()
    {
        return "y=sin(x)^2";
    }

    static float F(float x)
    {
        return sin(x) * sin(x);
    }

    // Indefinite integral from wolfram alpha
    // http://www.wolframalpha.com/input/?i=integrate+y%3Dsin(x)%5E2+from+0+to+pi
    static float IndefiniteIntegral(float x)
    {
        return x / 2.0f - sin(2.0f * x) / 4.0f;
    }
};

float Lerp(float a, float b, float t)
{
    return a * (1 - t) + b * t;
}

template <typename FUNCTION>
size_t MonteCarloIntegrate(float rangeMin, float rangeMax, float targetError)
{
    // get the actual answer
    float actualAnswer = FUNCTION::IndefiniteIntegral(rangeMax) - FUNCTION::IndefiniteIntegral(rangeMin);

    // integrate!
    float range = rangeMax - rangeMin;
    std::uniform_real_distribution<float> dist(rangeMin, rangeMax);
    size_t sampleCount = 0;
    float integration = 0.0f;
    do
    {
        ++sampleCount;
        float x = dist(g_mt);
        float estimate = range * FUNCTION::F(x);
        integration = Lerp(integration, estimate, 1.0f / float(sampleCount));
    }
    while (fabs(integration - actualAnswer) > targetError);

    // tell the caller how many samples it took
    return sampleCount;
}

template <typename FUNCTION, size_t NUM_TESTS>
void DoTests(float rangeMin, float rangeMax, float targetError)
{
    size_t min = 0;
    size_t max = 0;
    float average = 0.0f;

    for (size_t i = 1; i <= NUM_TESTS; ++i)
    {
        size_t sampleCount = MonteCarloIntegrate<FUNCTION>(rangeMin, rangeMax, targetError);

        if (i == 1 || sampleCount < min)
            min = sampleCount;
        
        if (i == 1 || sampleCount > max)
            max = sampleCount;

        average = Lerp(average, float(sampleCount), 1.0f / float(i));
    }

    float actualAnswer = FUNCTION::IndefiniteIntegral(rangeMax) - FUNCTION::IndefiniteIntegral(rangeMin);
    printf("%s from %f to %f = %f\n", FUNCTION::Name(), rangeMin, rangeMax, actualAnswer);
    printf("%zu runs, to get error <= %f\n", NUM_TESTS, targetError);
    printf("  min: %zu\n", min);
    printf("  max: %zu\n", max);
    printf("  avg: %f\n", average);
}

int main(int argc, char** argv)
{
    DoTests<SinX_Squared, 1000>(0.0f, c_pi, 0.0001f);

    system("pause");
    return 0;
}

/*

Example:
* integrate y=sin(x)^2 from 0 to pi.
* aka "What is the area under this graph?" Show the graph. probably the graph from the integral in wolfram alpha.

* probably should report variance or std deviation as part of the output

* I think it would be better to do N runs and then look at the error after that many runs
 * since you don't know the answer going into it, which is more realistic!


Blog:
1) explain using monte carlo to get the area under a curve
 * also have a program run that does it using random numbers.
 * make the program run until it's error is under a certain amount.
 * make it report how many samples it took.
 * do this N times, report average samples, min and max.
 * (could mention variance but meh)

2) Explain how if you control the shape of the random numbers you can make it converge faster
 * Do the same, but with RNG's that fit the function better.
 * the min / max / average should be better

3) Explain how if the shape doesn't match it can actually be wrong
 * demonstrate

* could show empiraclly how you need to quadruple to sample count to get half as much error!

* link to this code!

* likely will need to link to inversing CDF
 * and incremental averaging: https://blog.demofox.org/2016/08/23/incremental-averaging/

*/