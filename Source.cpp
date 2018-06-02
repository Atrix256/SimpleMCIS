#include <stdio.h>
#include <random>
#include <stdlib.h>

std::random_device g_rd;
std::mt19937 g_mt(g_rd());

static const float c_pi = 3.14159265359f;

// y = sin(x)^2
struct Function_SinX_Squared
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

struct PDF_Uniform
{
    static const char* Name()
    {
        return "Uniform Random Variables";
    }

    static float Generate(float rangeMin, float rangeMax)
    {
        std::uniform_real_distribution<float> dist(rangeMin, rangeMax);
        return dist(g_mt);
    }

    static float PDF(float x, float rangeMin, float rangeMax)
    {
        return 1.0f / (rangeMax - rangeMin);
    }
};

float Lerp(float a, float b, float t)
{
    return a * (1 - t) + b * t;
}

template <typename FUNCTION, typename PDF, size_t NUM_SAMPLES>
float MonteCarloIntegrate(float rangeMin, float rangeMax)
{
    // get the actual answer
    float actualAnswer = FUNCTION::IndefiniteIntegral(rangeMax) - FUNCTION::IndefiniteIntegral(rangeMin);

    // integrate!
    float integration = 0.0f;
    for (size_t i = 1; i <= NUM_SAMPLES; ++i)
    {
        float x = PDF::Generate(rangeMin, rangeMax);
        float pdf = PDF::PDF(x, rangeMin, rangeMax);
        float estimate = FUNCTION::F(x) / pdf;
        integration = Lerp(integration, estimate, 1.0f / float(i));
    }

    // tell the caller how much error there is at the end
    return fabs(integration - actualAnswer);
}

template <typename FUNCTION, typename PDF, size_t NUM_SAMPLES, size_t NUM_TESTS>
void DoTests(float rangeMin, float rangeMax)
{
    float minError = 0.0f;
    float maxError = 0.0f;
    float averageError = 0.0f;

    for (size_t i = 1; i <= NUM_TESTS; ++i)
    {
        float error = MonteCarloIntegrate<FUNCTION, PDF, NUM_SAMPLES>(rangeMin, rangeMax);

        if (i == 1 || error < minError)
            minError = error;
        
        if (i == 1 || error > maxError)
            maxError = error;

        averageError = Lerp(averageError, error, 1.0f / float(i));
    }

    float actualAnswer = FUNCTION::IndefiniteIntegral(rangeMax) - FUNCTION::IndefiniteIntegral(rangeMin);
    printf("Integrating %s using %s\n", FUNCTION::Name(), PDF::Name());
    printf("The actual integral of %s from %f to %f is %f\n", FUNCTION::Name(), rangeMin, rangeMax, actualAnswer);
    printf("Doing Monte Carlo integration...\n");
    printf("%zu runs, each taking %zu samples each run:\n", NUM_TESTS, NUM_SAMPLES);
    printf("  min Error: %f\n", minError);
    printf("  max Error: %f\n", maxError);
    printf("  avg Error: %f\n\n", averageError);
}

int main(int argc, char** argv)
{
    DoTests<Function_SinX_Squared, PDF_Uniform, 625, 1000>(0.0f, c_pi);
    DoTests<Function_SinX_Squared, PDF_Uniform, 2500, 1000>(0.0f, c_pi);
    DoTests<Function_SinX_Squared, PDF_Uniform, 10000, 1000>(0.0f, c_pi);
    DoTests<Function_SinX_Squared, PDF_Uniform, 40000, 1000>(0.0f, c_pi);

    // TODO: can we invert sin(x) and use that to generate randon numbers?

    // TODO: also use a PDF that has a different shape!

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
 * note this in the blog - met the error metric within 2-10 samples sometimes, even though the average was 20,000 and the maxes were up at like 1.5 million.
  * doesn't help though because you don't know the answer so don't know to stop! The next "wrong" answers will quickly push you away from being right ):


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
 * yep!

* link to this code!

* likely will need to link to inversing CDF
 * and incremental averaging: https://blog.demofox.org/2016/08/23/incremental-averaging/

*/