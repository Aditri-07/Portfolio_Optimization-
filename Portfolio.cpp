#include "Portfolio.h"
#include <nlopt.hpp>
#include <iostream>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <functional>
#include <stdexcept>

Portfolio::Portfolio(const std::vector<std::string>& assets, const std::vector<double>& weights)
    : assets(assets), weights(weights), expectedReturn(0.0), risk(0.0), VaR(0.0), AVaR(0.0) {}

void Portfolio::calculateExpectedReturn(const std::vector<double>& returns) {
    expectedReturn = std::inner_product(weights.begin(), weights.end(), returns.begin(), 0.0);
}

void Portfolio::calculateRisk(const std::vector<std::vector<double>>& covarianceMatrix) {
    risk = 0.0;
    for (size_t i = 0; i < weights.size(); ++i) {
        for (size_t j = 0; j < weights.size(); ++j) {
            risk += weights[i] * weights[j] * covarianceMatrix[i][j];
        }
    }
    risk = std::sqrt(risk);
}

// Value at Risk (VaR) 
double Portfolio::calculateVaR(double alpha) {
    double z_alpha = inverseCDF(1 - alpha); 
    VaR = expectedReturn - z_alpha * risk;
    return VaR;
}
//AVaR
double Portfolio::calculateAVaR(double alpha, const std::vector<double>& losses) {
    std::vector<double> sortedLosses = losses;
    std::sort(sortedLosses.begin(), sortedLosses.end());

    size_t VaRIndex = static_cast<size_t>((1 - alpha) * sortedLosses.size());
    double cumulativeLoss = std::accumulate(sortedLosses.begin() + VaRIndex, sortedLosses.end(), 0.0);
    size_t tailSize = sortedLosses.size() - VaRIndex;
    AVaR = cumulativeLoss / tailSize;
    return AVaR;
}

double Portfolio::inverseCDF(double alpha) {
    if (alpha <= 0.0 || alpha >= 1.0) {
        throw std::invalid_argument("Alpha must be between 0 and 1 for inverse CDF.");
    }

    // Abramowitz and Stegun approximation for the inverse CDF
    static const double a1 = -3.969683028665376e+01;
    static const double a2 = 2.209460984245205e+02;
    static const double a3 = -2.759285104469687e+02;
    static const double a4 = 1.383577518672690e+02;
    static const double a5 = -3.066479806614716e+01;
    static const double a6 = 2.506628277459239e+00;

    static const double b1 = -5.447609879822406e+01;
    static const double b2 = 1.615858368580409e+02;
    static const double b3 = -1.556989798598866e+02;
    static const double b4 = 6.680131188771972e+01;
    static const double b5 = -1.328068155288572e+01;

    static const double c1 = -7.784894002430293e-03;
    static const double c2 = -3.223964580411365e-01;
    static const double c3 = -2.400758277161838e+00;
    static const double c4 = -2.549732539343734e+00;
    static const double c5 = 4.374664141464968e+00;
    static const double c6 = 2.938163982698783e+00;

    static const double d1 = 7.784695709041462e-03;
    static const double d2 = 3.224671290700398e-01;
    static const double d3 = 2.445134137142996e+00;
    static const double d4 = 3.754408661907416e+00;

    double q = alpha < 0.5 ? alpha : 1.0 - alpha;
    double r;

    if (q > 0.02425) {
        r = q - 0.5;
        double u = r * r;
        return r * (((a1 * u + a2) * u + a3) * u + a4) * u + a5 /
               ((((b1 * u + b2) * u + b3) * u + b4) * u + b5) + a6;
    } else {
        r = std::sqrt(-2.0 * std::log(q));
        return ((c1 * r + c2) * r + c3) * r + c4 / (((d1 * r + d2) * r + d3) * r + d4);
    }
}

// Stochastic Dominance Constraints
double fsdConstraint(const std::vector<double>& x, std::vector<double>& grad, void* data) {
    auto* portfolio = static_cast<Portfolio*>(data);
    const std::vector<double>& benchmarkReturns = portfolio->getBenchmarkReturns();

    // Calculate VaR for the portfolio
    double portfolioVaR = portfolio->calculateVaR(0.05);

    if (std::isnan(portfolioVaR) || std::isinf(portfolioVaR)) {
        std::cerr << "Invalid VaR for portfolio" << std::endl;
        return std::numeric_limits<double>::infinity(); 
    }

    // Calculate VaR for the benchmark
    double benchmarkVaR = std::inner_product(benchmarkReturns.begin(), benchmarkReturns.end(), x.begin(), 0.0);

    if (std::isnan(benchmarkVaR) || std::isinf(benchmarkVaR)) {
        std::cerr << "Invalid VaR for benchmark" << std::endl;
        return std::numeric_limits<double>::infinity();
    }

    return portfolioVaR - benchmarkVaR; 
}

double ssdConstraint(const std::vector<double>& x, std::vector<double>& grad, void* data) {
    auto* portfolio = static_cast<Portfolio*>(data);
    const std::vector<std::vector<double>>& benchmarkLosses = portfolio->getBenchmarkLosses();


    double portfolioAVaR = portfolio->calculateAVaR(0.05, benchmarkLosses[0]);

    if (std::isnan(portfolioAVaR) || std::isinf(portfolioAVaR)) {
        std::cerr << "Invalid AVaR for portfolio" << std::endl;
        return std::numeric_limits<double>::infinity();
    }

    double benchmarkAVaR = portfolio->calculateAVaR(0.05, benchmarkLosses[1]);

    if (std::isnan(benchmarkAVaR) || std::isinf(benchmarkAVaR)) {
        std::cerr << "Invalid AVaR for benchmark" << std::endl;
        return std::numeric_limits<double>::infinity();
    }

    return portfolioAVaR - benchmarkAVaR; 
}

// Add Stochastic Dominance Constraints
void Portfolio::addStochasticDominanceConstraints(
    const std::vector<double>& benchmarkReturns, 
    double alpha, 
    const std::vector<std::vector<double>>& benchmarkLosses) {

    size_t n = weights.size();
    nlopt::opt opt(nlopt::LD_AUGLAG, n);

    opt.set_min_objective(
        [](const std::vector<double>& x, std::vector<double>& grad, void* data) {
            auto* covarianceMatrix = static_cast<std::vector<std::vector<double>>*>(data);
            double variance = 0.0;

            for (size_t i = 0; i < x.size(); ++i) {
                for (size_t j = 0; j < x.size(); ++j) {
                    variance += x[i] * x[j] * (*covarianceMatrix)[i][j];
                }
            }
            return variance;
        },
        const_cast<void*>(reinterpret_cast<const void*>(&benchmarkLosses))
    );

    opt.add_inequality_constraint(fsdConstraint, this, 1e-4);

    opt.add_inequality_constraint(ssdConstraint, this, 1e-4);

    opt.add_equality_constraint(
        [](const std::vector<double>& x, std::vector<double>& grad, void* data) {
            return std::accumulate(x.begin(), x.end(), 0.0) - 1.0;
        },
        nullptr, 1e-6
    );

    std::vector<double> lb(n, 0.0);
    std::vector<double> ub(n, 1.0);
    opt.set_lower_bounds(lb);
    opt.set_upper_bounds(ub);

    try {
        double minVariance;
        nlopt::result result = opt.optimize(weights, minVariance);
        std::cout << "Optimal weights with stochastic dominance constraints: ";
        for (double weight : weights) {
            std::cout << weight << " ";
        }
        std::cout << "\nMinimum variance: " << minVariance << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Stochastic dominance optimization failed: " << e.what() << std::endl;
    }
}

void Portfolio::updateRealTimeData(const std::vector<double>& updatedReturns,
                                    const std::vector<std::vector<double>>& updatedCovarianceMatrix) {
    calculateExpectedReturn(updatedReturns);
    calculateRisk(updatedCovarianceMatrix);
}

void Portfolio::optimizeEfficientFrontier(const std::vector<double>& returns,
                                          const std::vector<std::vector<double>>& covarianceMatrix) {
    size_t n = weights.size();
    nlopt::opt opt(nlopt::LD_AUGLAG, n);

    opt.set_min_objective(
        [](const std::vector<double>& x, std::vector<double>& grad, void* data) {
            auto* covarianceMatrix = static_cast<std::vector<std::vector<double>>*>(data);
            double variance = 0.0;
            for (size_t i = 0; i < x.size(); ++i) {
                for (size_t j = 0; j < x.size(); ++j) {
                    variance += x[i] * x[j] * (*covarianceMatrix)[i][j];
                }
            }
            return variance;
        },
        const_cast<void*>(reinterpret_cast<const void*>(&covarianceMatrix))
    );

    opt.add_equality_constraint(
        [](const std::vector<double>& x, std::vector<double>& grad, void* data) {
            return std::accumulate(x.begin(), x.end(), 0.0) - 1.0;
        },
        nullptr, 1e-8
    );

    opt.add_inequality_constraint(fsdConstraint, this, 1e-4);  

    opt.add_inequality_constraint(ssdConstraint, this, 1e-4);  


    std::vector<double> lb(n, 0.0);
    std::vector<double> ub(n, 1.0);
    opt.set_lower_bounds(lb);
    opt.set_upper_bounds(ub);

    double minVariance;
    try {
        nlopt::result result = opt.optimize(weights, minVariance);
        std::cout << "Optimal weights: ";
        for (double weight : weights) {
            std::cout << weight << " ";
        }
        std::cout << "\nMinimum variance: " << minVariance << std::endl;
    } catch (std::exception& e) {
        std::cerr << "Optimization failed: " << e.what() << std::endl;
    }
}

// portfolio details
void Portfolio::displayPortfolio() const {
    std::cout << "Portfolio:\n";
    for (size_t i = 0; i < assets.size(); ++i) {
        std::cout << assets[i] << ": " << weights[i] * 100 << "%\n";
    }
    std::cout << "Expected Return: " << expectedReturn << "\n";
    std::cout << "Risk (Std Dev): " << risk << "\n";
    std::cout << "VaR: " << VaR << "\n";
    std::cout << "AVaR: " << AVaR << "\n";
}
