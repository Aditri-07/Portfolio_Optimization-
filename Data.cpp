#include "Data.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <numeric>
#include <algorithm>
#include <cmath>
#include <stdexcept>

Data::Data() = default;

void Data::loadCSV(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file: " + filename);
    }

    std::string line;
    std::getline(file, line); // Skip header

    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string date, close;

        std::getline(ss, date, ',');   
        std::getline(ss, close, ','); 

        data["Date"].push_back(date);
        closePrices.push_back(std::stod(close));
    }
    file.close();
}

std::vector<double> Data::calculateReturns() {
    returns.clear();
    if (closePrices.size() < 2) {
        throw std::runtime_error("Insufficient data to calculate returns.");
    }

    for (size_t i = 1; i < closePrices.size(); ++i) {
        double ret = (closePrices[i] - closePrices[i - 1]) / closePrices[i - 1];
        returns.push_back(ret);
    }

    return returns;
}

double Data::calculateMean(const std::vector<double>& data) const {
    return std::accumulate(data.begin(), data.end(), 0.0) / data.size();
}

double Data::calculateVariance(const std::vector<double>& data) const {
    double mean = calculateMean(data);
    double variance = 0.0;
    for (double val : data) {
        variance += (val - mean) * (val - mean);
    }
    return variance / (data.size() - 1);
}

double Data::calculateCovariance(const std::vector<double>& data1, const std::vector<double>& data2) const {
    // both vectors must have the same size
    if (data1.size() != data2.size()) {
        throw std::invalid_argument("Data size mismatch for covariance calculation.");
    }

    double mean1 = calculateMean(data1);
    double mean2 = calculateMean(data2);

    double covariance = 0.0;
    for (size_t i = 0; i < data1.size(); ++i) {
        covariance += (data1[i] - mean1) * (data2[i] - mean2);
    }

    return covariance / data1.size();
}

double Data::calculateCorrelation(const std::vector<double>& data1, const std::vector<double>& data2) const {
    double covariance = calculateCovariance(data1, data2);
    double stdDev1 = std::sqrt(calculateVariance(data1));
    double stdDev2 = std::sqrt(calculateVariance(data2));
    return covariance / (stdDev1 * stdDev2);
}

const std::vector<double>& Data::getReturns() const {
    return returns;
}

const std::vector<double>& Data::getClosePrices() const {
    return closePrices;
}

void Data::displayData() const {
    std::cout << "Close Prices: " << closePrices.size() << " entries\n";
    std::cout << "Returns: " << returns.size() << " entries\n";

    for (size_t i = 0; i < closePrices.size(); ++i) {
        std::cout << "Close[" << i << "]: " << closePrices[i] << "\n";
    }
    for (size_t i = 0; i < returns.size(); ++i) {
        std::cout << "Return[" << i << "]: " << returns[i] << "\n";
    }
}

