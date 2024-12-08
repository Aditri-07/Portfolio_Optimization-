#ifndef DATA_H
#define DATA_H

#include <string>
#include <vector>
#include <map>

class Data {
private:
    std::map<std::string, std::vector<std::string>> data; 
    std::vector<double> closePrices;                     
    std::vector<double> returns;                          

public:
    Data();

    void loadCSV(const std::string& filename);
    std::vector<double> calculateReturns();

    double calculateMean(const std::vector<double>& data) const;
    double calculateVariance(const std::vector<double>& data) const;
    double calculateCovariance(const std::vector<double>& data1, const std::vector<double>& data2) const;
    double calculateCorrelation(const std::vector<double>& data1, const std::vector<double>& data2) const;

    const std::vector<double>& getReturns() const;
    const std::vector<double>& getClosePrices() const;

    void displayData() const;
};

#endif
