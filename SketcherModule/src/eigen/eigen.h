#include <iostream>

#include <Eigen/Dense>
#include <Eigen/Sparse>

#include <unsupported/Eigen/NonLinearOptimization>
//#include <unsupported/Eigen/LevenbergMarquardt>

// LM minimize for the model y = a x + b
typedef std::vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d> > Point2DVector;

// Generic functor
template<typename _Scalar, int NX = Eigen::Dynamic, int NY = Eigen::Dynamic>
struct Functor
{
    typedef _Scalar Scalar;
    enum
    {
        InputsAtCompileTime = NX,
        ValuesAtCompileTime = NY
    };
    typedef Eigen::Matrix<Scalar, InputsAtCompileTime, 1> InputType;
    typedef Eigen::Matrix<Scalar, ValuesAtCompileTime, 1> ValueType;
    typedef Eigen::Matrix<Scalar, ValuesAtCompileTime, InputsAtCompileTime> JacobianType;

    int m_inputs, m_values;

    Functor() : m_inputs(InputsAtCompileTime), m_values(ValuesAtCompileTime) {}
    Functor(int inputs, int values) : m_inputs(inputs), m_values(values) {}

    int inputs() const { return m_inputs; }
    int values() const { return m_values; }

};


struct MyFunctor : Functor<double>
{
    int operator()(const Eigen::VectorXd& x, Eigen::VectorXd& fvec) const
    {
        double a = x[0];
        double b = x[1];
        // "a" in the model is x(0), and "b" is x(1)
        for (unsigned int i = 0; i < this->Points.size(); ++i)
        {
            const auto& point = Points[i];
            fvec(i) = (double)((point(0) * point(0)) / (a * a) + (point(1) * point(1)) / (b * b) - 1.0);
        }

        return 0;
    }

    Point2DVector Points;

    int inputs() const { return 2; } // There are two parameters of the model
    int values() const { return this->Points.size(); } // The number of observations
};

struct MyFunctorNumericalDiff : Eigen::NumericalDiff<MyFunctor> {};

int main(int, char* [])
{
    unsigned int numberOfPoints = 50;
    Point2DVector points;

    Eigen::VectorXd x(2);
    x.fill(2.0f);

    MyFunctorNumericalDiff functor;
    functor.Points = points;
    Eigen::LevenbergMarquardt<MyFunctorNumericalDiff> lm(functor);

    Eigen::LevenbergMarquardtSpace::Status status = lm.minimize(x);
    std::cout << "status: " << status << std::endl;

    //std::cout << "info: " << lm.info() << std::endl;

    std::cout << "x that minimizes the function: " << std::endl << x << std::endl;

    return 0;
}