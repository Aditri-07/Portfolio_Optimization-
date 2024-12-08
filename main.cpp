#include "Portfolio.h"
#include "Data.h"
#include <iostream>
#include <vector>

int main() {
    try {

        Data nvidiaData;
        Data microsoftData;

        nvidiaData.loadCSV("nvidia.csv");
        microsoftData.loadCSV("microsoft.csv");

        nvidiaData.displayData();
        microsoftData.displayData();

        nvidiaData.calculateReturns();
        microsoftData.calculateReturns();

        size_t minSize = std::min(nvidiaData.getReturns().size(), microsoftData.getReturns().size());
        std::vector<double> nvidiaReturns(nvidiaData.getReturns().begin(), nvidiaData.getReturns().begin() + minSize);
        std::vector<double> microsoftReturns(microsoftData.getReturns().begin(), microsoftData.getReturns().begin() + minSize);

        double meanNVIDIA = nvidiaData.calculateMean(nvidiaReturns);
        double meanMicrosoft = microsoftData.calculateMean(microsoftReturns);

        double covariance = nvidiaData.calculateCovariance(nvidiaReturns, microsoftReturns);

        std::cout << "NVIDIA Mean Return: " << meanNVIDIA << "\n";
        std::cout << "Microsoft Mean Return: " << meanMicrosoft << "\n";
        std::cout << "Covariance between NVIDIA and Microsoft: " << covariance << "\n";

        std::vector<std::string> assets = {"NVIDIA", "Microsoft"};
        std::vector<double> initialWeights = {0.5, 0.5};

        Portfolio portfolio(assets, initialWeights);

        std::vector<double> portfolioReturns = {meanNVIDIA, meanMicrosoft};
        std::vector<std::vector<double>> covarianceMatrix = {
            {nvidiaData.calculateVariance(nvidiaReturns), covariance},
            {covariance, microsoftData.calculateVariance(microsoftReturns)}
        };

        portfolio.updateRealTimeData(portfolioReturns, covarianceMatrix);
        portfolio.calculateExpectedReturn(portfolioReturns);
        portfolio.calculateRisk(covarianceMatrix);

        double confidenceLevel = 0.95;
        portfolio.calculateVaR(1 - confidenceLevel);

        std::vector<double> combinedLosses = nvidiaData.getReturns(); 
        combinedLosses.insert(combinedLosses.end(), microsoftData.getReturns().begin(), microsoftData.getReturns().end());
        portfolio.calculateAVaR(1 - confidenceLevel, combinedLosses);

        portfolio.displayPortfolio();

        portfolio.optimizeEfficientFrontier(portfolioReturns, covarianceMatrix);

        std::vector<double> benchmarkReturns = {0.10, 0.12};
        std::vector<std::vector<double>> benchmarkLosses = {
            {-0.01, -0.02},
            {-0.03, -0.04}
        };
        portfolio.addStochasticDominanceConstraints(benchmarkReturns, confidenceLevel, benchmarkLosses);

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
    }

    return 0;
}
