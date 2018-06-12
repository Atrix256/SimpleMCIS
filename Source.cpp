#include <stdio.h>
#include <random>
#include <stdlib.h>

static const double c_pi = 3.14159265359;

// Note: the PDF's that were created were created specifically for this range.
// Changing these values without re-deriving the PDFs will give you wrong answers.
// You can use the PDFs for new functions on the same interval though.
static const double c_rangeMin = 0.0f;
static const double c_rangeMax = c_pi; 

std::random_device g_rd;
std::mt19937 g_mt(g_rd());
std::uniform_real_distribution<double> g_dist_0_1(0.0, 1.0);

static const size_t c_numSamples = 50 * 1000 * 1000;

// y = sin(x)^2
struct Function_SinX_Squared
{
    static const char* Name()
    {
        return "y=sin(x)^2";
    }

    static double F(double x)
    {
        return sin(x) * sin(x);
    }

    // Indefinite integral from wolfram alpha
    // http://www.wolframalpha.com/input/?i=integrate+y%3Dsin(x)%5E2+from+0+to+pi
    static double IndefiniteIntegral(double x)
    {
        return x / 2.0 - sin(2.0 * x) / 4.0;
    }
};

// y = sin(x)
struct Function_SinX
{
    static const char* Name()
    {
        return "y=sin(x)";
    }

    static double F(double x)
    {
        return sin(x);
    }

    // Indefinite integral from wolfram alpha
    // http://www.wolframalpha.com/input/?i=integrate+y%3Dsin(x)+from+0+to+pi
    static double IndefiniteIntegral(double x)
    {
        return -cos(x);
    }
};

struct PDF_Uniform
{
    static const char* Name()
    {
        return "PDF y=1/pi";
    }

    static double InverseCDF(double rnd)
    {
        return rnd * c_pi;
    }

    static double PDF(double x)
    {
        return 1.0 / c_pi;
    }
};

struct PDF_SinX
{
    static const char* Name()
    {
        return "PDF y=sin(x)/2";
    }

    static double InverseCDF(double rnd)
    {
        return 2.0 * asin(sqrt(rnd));
    }

    static double PDF(double x)
    {
        return sin(x) / 2.0;
    }
};

struct PDF_CosXOver2Over2
{
    static const char* Name()
    {
        return "PDF y=cos(x/2)/2";
    }

    static double InverseCDF(double rnd)
    {
        return 2.0 * asin(rnd);
    }

    static double PDF(double x)
    {
        return cos(x/2.0) / 2.0;
    }
};

struct PDF_XOverPiToTheFifth
{
    static const char* Name()
    {
        return "PDF y=(x/pi)^5 * 6.0 / pi";
    }

    static double InverseCDF(double rnd)
    {
        return pow(rnd * pow(c_pi, 6.0), 1.0 / 6.0);
    }

    static double PDF(double x)
    {
        return pow(x / c_pi, 5.0) * 6.0 / c_pi;
    }
};

struct PDF_XOverPiToTheSecond
{
    static const char* Name()
    {
        return "PDF y=(x/pi)^2 * 3.0 / pi";
    }

    static double InverseCDF(double rnd)
    {
        return pow(rnd * pow(c_pi, 3.0), 1.0 / 3.0);
    }

    static double PDF(double x)
    {
        return pow(x / c_pi, 2.0) * 3.0 / c_pi;
    }
};

template <typename T>
T Lerp(T a, T b, T t)
{
    return a * (T(1) - t) + b * t;
}

double SimpleMonteCarlo()
{
    double rangeMin = 0;
    double rangeMax = 3.14159265359;

    size_t numSamples = 10000;

    std::random_device rd;
    std::mt19937 mt(rd());
    std::uniform_real_distribution<double> dist(rangeMin, rangeMax);

    double ySum = 0.0;
    for (size_t i = 1; i <= numSamples; ++i)
    {
        double x = dist(mt);
        double y = sin(x)*sin(x);
        ySum += y;
    }
    double yAverage = ySum / double(numSamples);

    double width = rangeMax - rangeMin;
    double height = yAverage;

    return width * height;
}

double GeneralMonteCarlo()
{
    size_t numSamples = 10000;

    std::random_device rd;
    std::mt19937 mt(rd());
    std::uniform_real_distribution<double> dist(0.0f, 1.0f);

    auto InverseCDF = [](double x) -> double
    {
        return x * c_pi;
    };

    auto PDF = [](double x) -> double
    {
        return 1.0f / c_pi;
    };

    double estimateSum = 0.0;
    for (size_t i = 1; i <= numSamples; ++i)
    {
        double rnd = dist(mt);
        double x = InverseCDF(rnd);
        double y = sin(x)*sin(x);
        double pdf = PDF(x);
        double estimate = y / pdf;

        estimateSum += estimate;
    }
    double estimateAverage = estimateSum / double(numSamples);

    return estimateAverage;
}

double ImportanceSampledMonteCarlo()
{
    size_t numSamples = 10000;

    std::random_device rd;
    std::mt19937 mt(rd());
    std::uniform_real_distribution<double> dist(0.0, 1.0);

    auto InverseCDF = [](double x) -> double
    {
        return 2.0 * asin(sqrt(x));
    };

    auto PDF = [](double x) -> double
    {
        return sin(x) / 2.0f;
    };

    double estimateSum = 0.0;
    for (size_t i = 1; i <= numSamples; ++i)
    {
        double rng = dist(mt);
        double x = InverseCDF(rng);
        double y = sin(x)*sin(x);
        double pdf = PDF(x);
        double estimate = y / pdf;

        estimateSum += estimate;
    }
    double estimateAverage = estimateSum / double(numSamples);

    return estimateAverage;
}

template <typename FUNCTION>
void Test_MC()
{
    double actualAnswer = FUNCTION::IndefiniteIntegral(c_rangeMax) - FUNCTION::IndefiniteIntegral(c_rangeMin);

    printf("Integrating %s from %f to %f\nThe actual answer is %f\n", FUNCTION::Name(), c_rangeMin, c_rangeMax, actualAnswer);
    printf("Doing Monte Carlo integration with %zu samples:\n", c_numSamples);

    std::uniform_real_distribution<double> dist(c_rangeMin, c_rangeMax);

    double range = c_rangeMax - c_rangeMin;
    double integration = 0.0;
    double averageDifferenceSquared = 0.0;
    for (size_t i = 1; i <= c_numSamples; ++i)
    {
        // integrate
        double x = dist(g_mt);
        double estimate = FUNCTION::F(x) * range;
        integration = Lerp(integration, estimate, 1.0 / double(i));

        // Variance is "The average of the squared differences from the mean"
        double difference = integration - actualAnswer;
        double differenceSquared = difference * difference;
        averageDifferenceSquared = Lerp(averageDifferenceSquared, differenceSquared, 1.0 / double(i));

        // report progress at specific progress points
        if (i == 1 || 
            i == c_numSamples / 4096 ||
            i == c_numSamples / 1024 ||
            i == c_numSamples / 256 ||
            i == c_numSamples / 64 ||
            i == c_numSamples / 16 ||
            i == c_numSamples / 4 ||
            i == c_numSamples)
        {
            double difference = integration - actualAnswer;
            printf("[%10.zu] %f  (%s%f) (estimate stddev: %f)\n", i, integration, difference >= 0 ? "+" : "", difference, sqrt(averageDifferenceSquared));
        }
    }
    printf("\n");
}

template <typename FUNCTION, typename PDF>
void Test_MC_PDF()
{
    double actualAnswer = FUNCTION::IndefiniteIntegral(c_rangeMax) - FUNCTION::IndefiniteIntegral(c_rangeMin);

    printf("Integrating %s from %f to %f\nThe actual answer is %f\n", FUNCTION::Name(), c_rangeMin, c_rangeMax, actualAnswer);
    printf("Doing Monte Carlo integration with %zu samples, using %s.\nEstimates:\n", c_numSamples, PDF::Name());

    double integration = 0.0f;
    double averageDifferenceSquared = 0.0;
    for (size_t i = 1; i <= c_numSamples; ++i)
    {
        // integrate
        double rnd = g_dist_0_1(g_mt);
        double x = PDF::InverseCDF(rnd);
        double pdf = PDF::PDF(x);
        double estimate = FUNCTION::F(x) / pdf;
        integration = Lerp(integration, estimate, 1.0 / double(i));

        // Variance is "The average of the squared differences from the mean"
        double difference = integration - actualAnswer;
        double differenceSquared = difference * difference;
        averageDifferenceSquared = Lerp(averageDifferenceSquared, differenceSquared, 1.0 / double(i));

        // report progress at specific progress points
        if (i == 1 ||
            i == c_numSamples / 4096 ||
            i == c_numSamples / 1024 ||
            i == c_numSamples / 256 ||
            i == c_numSamples / 64 ||
            i == c_numSamples / 16 ||
            i == c_numSamples / 4 ||
            i == c_numSamples)
        {
            double difference = integration - actualAnswer;
            printf("[%10.zu] %f  (%s%f) (estimate stddev: %f)\n", i, integration, difference >= 0 ? "+" : "", difference, sqrt(averageDifferenceSquared));
        }
    }
    printf("\n");
}

int main(int argc, char** argv)
{
    printf("Simple Monte Carlo says: %f\n\n", SimpleMonteCarlo());
    printf("General Monte Carlo says: %f\n\n", GeneralMonteCarlo());
    printf("Importance Sampled Monte Carlo says: %f\n\n", ImportanceSampledMonteCarlo());

    Test_MC<Function_SinX_Squared>();
    Test_MC_PDF<Function_SinX_Squared, PDF_Uniform>();
    Test_MC_PDF<Function_SinX_Squared, PDF_SinX>();
    Test_MC_PDF<Function_SinX_Squared, PDF_CosXOver2Over2>();
    Test_MC_PDF<Function_SinX_Squared, PDF_XOverPiToTheFifth>();
    Test_MC_PDF<Function_SinX_Squared, PDF_XOverPiToTheSecond>();

    Test_MC_PDF<Function_SinX, PDF_Uniform>();
    Test_MC_PDF<Function_SinX, PDF_SinX>();



    /*
    Process New - for blog

    TODO: put the derivations in a special section at the end so the post doesn't get all clogged up with math
    TODO: quadratic fit (stuck) - could try a cubic fit on inverted CDF data
    TODO: sin(x)^2 (stuck)
    TODO: cos(x) (should be easy to do)


    ----- (x/pi)^2 -----

    F(x) = (x/pi)^2

    Integrate that and we get: G(x) = x^3 / (3*pi^2)

    We can get the normalization constant for the PDF by taking G(pi) - G(0). Do that and you get: pi/3

    So, PDF(x) = (x/pi)^2 * 3/pi

    To make G be the CDF so that we can invert it, we need to make sure that CDF(0) is 0 and CDF(pi) is 1.

    So we do this:

    H(x) = G(x) - G(0)
    CDF(x) = H(x) / H(pi)

    H(x) = x^3 / (3*pi^2)
    CDF(x) = x^3 / pi^3
    
    looks decent: http://www.wolframalpha.com/input/?i=x%5E3+%2F+pi%5E3+from+0+to+pi

    Now to invert the CDF we flip y and X and solve for y again

    CDF^-1(x) = (y * pi^3)^(1/3)


    ----- (x/pi)^5 -----

    F(x) = (x/pi)^5

    Integrate that and we get: G(x) = x^6 / (6*pi^5)

    We can get the normalization constant for the PDF by taking G(pi) - G(0). Do that and you get: pi/6

    So, PDF(x) = (x/pi)^5 * 6/pi

    To make G be the CDF so that we can invert it, we need to make sure that CDF(0) is 0 and CDF(pi) is 1.

    So we do this:

    H(x) = G(x) - G(0)
    CDF(x) = H(x) / H(pi)

    H(X) = x^6 / (6*pi^5)
    CDF(x) = x^6 / pi^6

    looks decent: http://www.wolframalpha.com/input/?i=x%5E6+%2F+pi%5E6+from+0+to+pi

    Now to invert the CDF we flip y and X and solve for y again

    CDF^-1(x) = (y * pi^6)^(1/6)

    looks legit: http://www.wolframalpha.com/input/?i=(y+*+pi%5E6)%5E(1%2F6)+from+0+to+1


    ----- cos(x/2) PDF -----

    F(x) = cos(x/2)

    Integrate that and we get: G(x) = 2 * sin(x/2)

    We can get the normalization constant for the PDF by taking G(pi) - G(0). Do that and you get: 2

    So, PDF(x) = cos(x/2) / 2

    To make G be the CDF so that we can invert it, we need to make sure that CDF(0) is 0 and CDF(pi) is 1.

    So we do this:

    H(x) = G(x) - G(0)
    CDF(x) = H(x) / H(pi)

    H(x) = 2*sin(x/2)
    CDF(x) = sin(x/2)

    looks good: http://www.wolframalpha.com/input/?i=sin(x%2F2)+from+0+to+pi

    Now to invert the CDF we flip y and X and solve for y again

    CDF^-1 is
    y = 2 * (2 * pi * n - sin^(-1)(x) + pi) and n in Z

    actually the other answer...
    y = 2 (2 pi n + sin^(-1)(x)) and n element Z

    Yep!

    y = 2 * sin^(-1)(x)

    !!! convergence isn't so different. need to find a better function.

    ----- 1-Sin(x) PDF -----

    F(x) = 1 - sin(x)

    Integrate that and we get: G(x) = x+cos(x)

    We can get the normalization constant for the PDF by taking G(pi) - G(0). Do that and you get pi - 2

    So, PDF(x) = (1-sin(x)) / (pi-2)

    To make G be the CDF so that we can invert it, we need to make sure that CDF(0) is 0 and CDF(pi) is 1.

    So we do this:

    H(x) = G(x) - G(0)
    CDF(x) = H(x) / H(pi)

    H(x) = x+cos(x)-1
    CDF(x) = (x+cos(x)-1) / (pi-2)

    looks good: http://www.wolframalpha.com/input/?i=graph+(x%2Bcos(x)-1)+%2F+(pi-2)+from+0+to+pi

    Now to invert the CDF we flip y and X and solve for y again

    !!! CRAP. x+cos(x) not invertible, i'm stuck.

    ----- Sin(x)^2 PDF -----

    What if we use a PDF that exactly matches the function we are integrating?! (NOTE: this is what cosine weighted hemispher sampling does!)

    F(x) = sin(x)^2

    Integrate that and we get: G(x) = 1/2 (x - sin(x)*cos(x))

    We can get the normalization constant for the PDF by taking G(pi) - G(0). Do that and you get pi / 2

    So, PDF(x) = sin(x)^2 * 2 / pi

    To make G be the CDF so that we can invert it, we need to make sure that CDF(0) is 0 and CDF(pi) is 1.

    So we do this:

    H(x) = G(x) - G(0)
    CDF(x) = H(x) / H(pi)

    H(x) = 1/2 (x - sin(x)*cos(x))
    CDF(x) = (x - sin(x)*cos(x)) / pi

    looks legit: http://www.wolframalpha.com/input/?i=graph+y+%3D(x+-+sin(x)*cos(x))+%2F+pi+from+0+to+pi

    Now to invert the CDF we flip y and X and solve for y again

    It's a bit scary to invert, but luckily there's this:

    "The double-angle identity might help you a bit. sin(2theta) = 2sin(theta)cos(theta)"
    https://twitter.com/scottmichaud/status/1003033402411544577

    So, an alternate form of the CDF is...
    CDF(x) = (2x - sin(2x)) / (2 * pi)

    TODO: continue here, almost there, if you can get that thing inverted!



    ----- Sin(x) PDF -----

    We want something roughly the shape of the thing we are integrating - how bout sin(x)?

    F(x) = sin(x)

    Show graph of this, and what we are integrating. This is the un-normalized PDF.

    Integrate that and we get: G(x) = -cos(x)

    We can get the normalization constant for the PDF by taking G(pi) - G(0). Do that and you get 2.
    
    So, PDF(x) = sin(x) / 2

    To make G be the CDF so that we can invert it, we need to make sure that CDF(0) is 0 and CDF(pi) is 1.

    So we do this:

    H(x) = G(x) - G(0)
    CDF(x) = H(x) / H(pi)

    H(X) = -cos(x) + 1
    CDF(x) = (-cos(x) + 1) / (-cos(pi) + 1)

    Looks reasonable: http://www.wolframalpha.com/input/?i=(-cos(x)+%2B+1)+%2F+(-cos(pi)+%2B+1)+from+0+to+pi

    Now to invert the CDF we flip y and X and solve for y again

    http://www.wolframalpha.com/input/?i=x+%3D+(-cos(y)+%2B+1)+%2F+(-cos(pi)+%2B+1)+solve+for+y

    Gets a bit confusing but...

    InverseCDF = 2 (sin^(-1)(sqrt(x)))



    ----- Quadratic curve pdf -----
    
    We want something roughly the shape of the thing we are integrating - how bout a quadratic?

    Fit a curve to: (0,0), (pi/2, sin(pi/2)^2), (pi, 0)

    Wolfram alpha: http://www.wolframalpha.com/input/?i=quadratic+fit+%7B%7B0,0%7D,%7Bpi%2F2,1%7D,%7Bpi,0%7D%7D
    Info about curve fitting: https://blog.demofox.org/2016/12/22/incremental-least-squares-curve-fitting/

    Gives: F(x) = (4x) / (pi) - (4*x^2) / (pi^2)

    Show graph of this, and what we are integrating. This is the un-normalized PDF.

    Integrate that and we get: G(x) = (2x^2) / (pi) - (4x^3) / (3 * pi^2)

    We can get the normalization constant for the PDF by taking G(pi) - G(0) which you can plug pi in for x and get (2*pi)/3.
    We divide F by that value to make the area under the curve be 1 which is a requirement of a function being a PDF.

    so, PDF(x) = ((4x) / (pi) - (4*x^2) / (pi^2)) * 3 / (2*pi)

    (can verify: http://www.wolframalpha.com/input/?i=integrate+y+%3D+((4x)+%2F+(pi)+-+(4*x%5E2)+%2F+(pi%5E2))+*+3+%2F+(2*pi)+from+0+to+pi)

    To make G be the CDF so that we can invert it, we need to make sure that CDF(0) is 0 and CDF(pi) is 1.

    So we do this:

    H(x) = G(x) - G(0)
    CDF(x) = H(x) / H(pi)

    And we get:

    CDF(x) = ((2x^2) / (pi) - (4x^3) / (3 * pi^2)) * 3 / (2*pi)

    Now to invert the CDF we flip y and X and solve for y again

    !! A mess with 3 answers: http://www.wolframalpha.com/input/?i=x+%3D+((2y%5E2)+%2F+(pi)+-+(4y%5E3)+%2F+(3+*+pi%5E2))+*+3+%2F+(2*pi)+solve+for+y

    
    */




    // BETTER...
    // sin(x) looks similar (show graph)
    // make CDF:
    //  integral of sin(x) = -cos(x)
    //  need to add -cos(0) which is 1: y = -cos(x) + 1
    //  TODO: need to normalize it (integrate from 0 to pi and divide that value)
    //  TODO: then need to invert it.
    //  TODO: make PDF object have functions on it...
    //   1) PDF(x) -> probability of choosing x
    //   2) InvertedCDF(x) -> turn a randomly selected random number into a number from the PDF distribution

    // TODO: maybe make a "DoTestsUniform" (simple) and a "DoTestsPDF". show equivelance by doing same uniform tests with PDF before moving on.


    // TODO: can we invert sin(x) and use that to generate randon numbers?
    // Notes for blog...
    // http://www.wolframalpha.com/input/?i=y%3Dsin%5E-1(x)
    // Or, if not, invert a quadratic maybe? where the two zeros are 0 and pi, and the hump goes up.
    // Maybe do a quadratic curve fit to the points...
    // (0,0)
    // (pi/2, sin^2(pi/2)) aka (pi/2, 1)
    // (pi,0)
    // seems like this does it: y=-0.41x^2 + 1.27x
    // via http://demofox.org/LeastSquaresCurveFit.html
    // put in google for graphs: graph y = -0.41x^2 + 1.27x, y = sin(x)^2 from 0 to pi
    // Invert quadratic function by flipping x and y and solving for y again (link to invert pdf)
    // TODO: erm... need to turn pdf into cdf before inverting it! maybe try with sin(x) again?
    // Wolfram: http://www.wolframalpha.com/input/?rawformassumption=%22ClashPrefs%22+-%3E+%7B%22Math%22%7D&i=x+%3D+-0.41y%5E2+%2B+1.27y+solve+for+y
    // gives: y = 1/82 * (127 - sqrt(16129 - 16400x))
    // and (???) : y = 1/82 * (sqrt(16129 - 16400x) + 127)

    // TODO: also use a PDF that has a different shape!

    system("pause");
    return 0;
}

/*
! it could be fun to make a program where you give it a function and it gives you a curve fit based PDF and inverted CDF
 ? how would that work exactly
 * if curve fitting one, it would be nice to generate the other as a complement to it, so there isn't two fudge amounts there.

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
 * and least squares fit...
  * http://demofox.org/LeastSquaresCurveFit.html
  * https://blog.demofox.org/2016/12/22/incremental-least-squares-curve-fitting/




Example:
1) Average height * width
2) uniform PDF (same results)
3) quadratic good PDF
4) sin PDF
5) sin^2 PDF (exact!)
6) cos^2 PDF (bad)

 
*/