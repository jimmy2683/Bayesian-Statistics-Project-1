#include<iostream>
#include<fstream>
#include<vector>
#include<string>
#include<sstream>
#include<algorithm>
#include<cmath>
#include<numeric>

using namespace std;

struct DailyData {
    string date;
    double price;
};

// Function to calculate skewness
double calculate_skewness(const vector<double>& data, double mean, double stddev) {
    if (stddev == 0) return 0;
    double skew = 0.0;
    for (double val : data) {
        skew += pow((val - mean) / stddev, 3);
    }
    return skew / data.size();
}

// Function to calculate kurtosis
double calculate_kurtosis(const vector<double>& data, double mean, double stddev) {
    if (stddev == 0) return 0;
    double kurt = 0.0;
    for (double val : data) {
        kurt += pow((val - mean) / stddev, 4);
    }
    return (kurt / data.size()) - 3.0; // Excess kurtosis
}

int main(){
	
	ifstream data("data.csv");
    if (!data.is_open()) {
        cerr << "Error opening data.csv" << endl;
        return 1;
    }

    vector<DailyData> records;
    string line;

    // Skip header lines
    getline(data, line);
    getline(data, line);

    while(getline(data, line)) {
        stringstream ss(line);
        string field;
        
        // leading comma
        getline(ss, field, ',');
        if (field.empty() && ss.eof()) continue;

        DailyData record;
        
        // Date
        getline(ss, field, '"'); // consume until first quote
        getline(ss, record.date, '"'); // read date
        getline(ss, field, ','); // consume comma after date

        // Price
        getline(ss, field, ',');
        try {
            record.price = stod(field);
            records.push_back(record);
        } catch (const std::invalid_argument& ia) {
            // Ignore lines that can't be parsed
        }
    }
    data.close();

    // Data is in reverse chronological order, so reverse it
    reverse(records.begin(), records.end());

    // --- Price Trend ---
    ofstream prices_out("prices.csv");
    prices_out << "Date,Price\n";
    for(const auto& record : records) {
        prices_out << '"' << record.date << "\"," << record.price << "\n";
    }
    prices_out.close();

    // --- Returns Computation ---
    vector<double> log_returns;
    for(size_t i = 1; i < records.size(); ++i) {
        if (records[i-1].price > 0) {
            log_returns.push_back(log(records[i].price / records[i-1].price));
        }
    }

    ofstream returns_out("returns.csv");
    returns_out << "LogReturn\n";
    for(double ret : log_returns) {
        returns_out << ret << "\n";
    }
    returns_out.close();

    // Calculate stats for log returns
    double sum = accumulate(log_returns.begin(), log_returns.end(), 0.0);
    double mean = sum / log_returns.size();
    double sq_sum = inner_product(log_returns.begin(), log_returns.end(), log_returns.begin(), 0.0);
    double variance = (sq_sum / log_returns.size()) - mean * mean;
    double stddev = sqrt(variance);
    double skewness = calculate_skewness(log_returns, mean, stddev);
    double kurtosis = calculate_kurtosis(log_returns, mean, stddev);

    cout << "Log Returns Statistics:" << endl;
    cout << "Mean: " << mean << endl;
    cout << "Variance: " << variance << endl;
    cout << "Skewness: " << skewness << endl;
    cout << "Kurtosis: " << kurtosis << endl;

    // --- Volatility Analysis (Rolling Mean & StdDev) ---
    int window_size = 20;
    ofstream rolling_out("rolling_stats.csv");
    rolling_out << "Date,Price,RollingMean,RollingStd\n";
    for(size_t i = 0; i < records.size(); ++i) {
        rolling_out << '"' << records[i].date << "\"," << records[i].price;
        if (i >= window_size - 1) {
            double rolling_sum = 0.0;
            for(int j = 0; j < window_size; ++j) {
                rolling_sum += records[i-j].price;
            }
            double rolling_mean = rolling_sum / window_size;

            double rolling_sq_sum = 0.0;
            for(int j = 0; j < window_size; ++j) {
                rolling_sq_sum += pow(records[i-j].price - rolling_mean, 2);
            }
            double rolling_std = sqrt(rolling_sq_sum / window_size);
            rolling_out << "," << rolling_mean << "," << rolling_std;
        } else {
            rolling_out << ",,"; // No value for first entries
        }
        rolling_out << "\n";
    }
    rolling_out.close();
		
	return 0;
}
