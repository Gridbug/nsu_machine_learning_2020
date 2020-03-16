#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>

#include <QDataStream>
#include <QFile>
#include <QTextStream>

using MyDataType = std::vector<std::pair<std::string, std::vector<double>>>;

struct WaterPlantDataEntry {
    std::string day;
    std::vector<double> features;
    int32_t clusterId = 0;
};

std::vector<std::vector<double>> initialCentroids(std::vector<WaterPlantDataEntry> data, int32_t hyperparameterK);
double euclideanDistance(std::vector<double> a, std::vector<double> b);

int main(int argc, char* argv[]) {
    if (3 != argc) {
        std::cerr << "ERROR: Wrong number of arguments. Usage " << argv[0] << " <data_file_name> <hyperparameter_k>" << std::endl;
        return -1;
    }

    QFile dataFile(argv[1]);

    dataFile.open(QFile::ReadOnly);
    if (!dataFile.isOpen()) {
        std::cerr << "ERROR: Failed to open data file \"" << argv[1] << "\"" << std::endl;
        return -1;
    }

    QTextStream dataStream(&dataFile);

    const uint32_t numFeatures = 38;
    std::vector<WaterPlantDataEntry> data;

    while (!dataStream.atEnd()) {
        QString currentLine = dataStream.readLine();
        if (currentLine.isEmpty()) {
            continue;
        }

        auto splittedLine = currentLine.split(",");
        if (splittedLine.length() != numFeatures + 1)  {
            std::cerr << "WARNING: Skipping line with not enough data. Found only " << splittedLine.length() << " comma separated values, expected " << numFeatures + 1 << std::endl;
            continue;
        }

        std::vector<double> numericalData;

        std::transform(splittedLine.begin() + 1,
                       splittedLine.end(),
                       std::back_inserter(numericalData),
                       [] (QString dataStr) {
                            return dataStr.toDouble();
                       });

        data.push_back({splittedLine.first().toStdString(), numericalData});
    }

    dataFile.close();

    if (data.empty()) {
        std::cerr << "ERROR: Not found any data in file \"" << argv[1] << "\"" << std::endl;
        return -1;
    }

    std::cout << "INFO: Found " << data.size() << " data lines and successfully parsed them." << std::endl;

    const uint32_t hyperparameterK = std::stoi(argv[2]);

    std::vector<std::vector<double>> centroids = initialCentroids(data, hyperparameterK);

    for (auto& dataEntry : data) {
        std::vector<double> distancesToCentroids;

        for (auto centroidFeatures : centroids) {
            distancesToCentroids.push_back(euclideanDistance(dataEntry.features, centroidFeatures));
        }

        auto minIt = std::min_element(distancesToCentroids.begin(), distancesToCentroids.end());

        dataEntry.clusterId = std::distance(distancesToCentroids.begin(), minIt);
    }

    return 0;
}

std::vector<std::vector<double>> initialCentroids(std::vector<WaterPlantDataEntry> data, int32_t hyperparameterK) {
    std::vector<std::vector<double>> centroids(hyperparameterK);

    for (int32_t i = 0; i < hyperparameterK; i++) {
        centroids[i] = data[i].features;
    }

    return centroids;
}

double euclideanDistance(std::vector<double> a, std::vector<double> b) {
    double distance = 0;

    for (uint32_t featureId = 0; featureId < a.size(); featureId++) {
        distance += pow(a[featureId] - b[featureId], 2);
    }

    return sqrt(distance);
}
