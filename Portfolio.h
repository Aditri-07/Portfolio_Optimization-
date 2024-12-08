#ifndef PORTFOLIO_H
#define PORTFOLIO_H

#include <vector>
#include <string>

class Portfolio {
private:
    std::vector<std::string> assets;
    std::vector<double> weights;
    double expectedReturn;
    double risk;
    double VaR;
    double AVaR;

    std::vector<double> benchmarkReturns; 
    std::vector<std::vector<double>> benchmarkLosses; 

public:
    Portfolio(const std::vector<std::string>& assets, const std::vector<double>& weights);

    void calculateExpectedReturn(const std::vector<double>& returns);
    void calculateRisk(const std::vector<std::vector<double>>& covarianceMatrix);
    double calculateVaR(double alpha);
    double calculateAVaR(double alpha, const std::vector<double>& losses);

    // inverse CDF for normal distribution
    double inverseCDF(double alpha);

    const std::vector<double>& getBenchmarkReturns() const { return benchmarkReturns; }
    const std::vector<std::vector<double>>& getBenchmarkLosses() const { return benchmarkLosses; }

    void optimizeEfficientFrontier(const std::vector<double>& returns, 
                                   const std::vector<std::vector<double>>& covarianceMatrix);
    
    void addStochasticDominanceConstraints(
        const std::vector<double>& benchmarkReturns, 
        double alpha, 
        const std::vector<std::vector<double>>& benchmarkLosses);

    void displayPortfolio() const;

    void updateRealTimeData(const std::vector<double>& updatedReturns,
                            const std::vector<std::vector<double>>& updatedCovarianceMatrix);

    // getters
    double getExpectedReturn() const { return expectedReturn; }
    double getRisk() const { return risk; }
};

#endif
