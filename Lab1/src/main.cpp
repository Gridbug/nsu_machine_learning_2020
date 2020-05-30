#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <set>
#include <limits>

#include <QDataStream>
#include <QFile>
#include <QTextStream>

using MyDataType = std::vector<std::pair<std::string, std::vector<double>>>;

struct WaterPlantDataEntry {
    std::string day;
    std::vector<double> features;
    int32_t clusterId = 0;
};

const uint32_t maxNumIterations = 100;
const uint32_t numFeatures = 38;

std::vector<std::vector<double>> initialCentroids(std::vector<WaterPlantDataEntry> data, int32_t hyperparameterK);
std::vector<std::vector<double>> randInitialCentroids(std::vector<WaterPlantDataEntry> data, int32_t hyperparameterK);
double euclideanDistance(std::vector<double> a, std::vector<double> b);
std::vector<std::vector<double>> computeCentroids(std::vector<WaterPlantDataEntry> data, int32_t hyperparameterK);
void fixMissingData(std::vector<WaterPlantDataEntry>& data);
void dataNormalization(std::vector<WaterPlantDataEntry>& data);

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
                            bool status = true;
                            double convertedValue = dataStr.toDouble(&status);
                            if (status) {
                                return convertedValue;
                            } else {
                                return std::numeric_limits<double>::quiet_NaN();
                            }
                       });

        data.push_back({splittedLine.first().toStdString(), numericalData});
    }

    dataFile.close();

    fixMissingData(data);

    dataNormalization(data);

    srand(time(nullptr));

    if (data.empty()) {
        std::cerr << "ERROR: Not found any data in file \"" << argv[1] << "\"" << std::endl;
        return -1;
    }

    std::cout << "INFO: Found " << data.size() << " data lines and successfully parsed them." << std::endl;

    const uint32_t hyperparameterK = std::stoi(argv[2]);

    std::vector<std::vector<double>> centroids = randInitialCentroids(data, hyperparameterK);

    for (auto& dataEntry : data) {
        std::vector<double> distancesToCentroids;

        for (auto centroidFeatures : centroids) {
            distancesToCentroids.push_back(euclideanDistance(dataEntry.features, centroidFeatures));
        }

        auto minIt = std::min_element(distancesToCentroids.begin(), distancesToCentroids.end());

        dataEntry.clusterId = std::distance(distancesToCentroids.begin(), minIt);
    }

    uint32_t i;
    for (i = 0; i < maxNumIterations; i++) {
        centroids = computeCentroids(data, hyperparameterK);

        bool nothingChanged = true;

        for (auto& dataEntry : data) {
            std::vector<double> distancesToCentroids;

            for (auto centroidFeatures : centroids) {
                distancesToCentroids.push_back(euclideanDistance(dataEntry.features, centroidFeatures));
            }

            auto minIt = std::min_element(distancesToCentroids.begin(), distancesToCentroids.end());

            auto newClusterId = std::distance(distancesToCentroids.begin(), minIt);

            if (newClusterId == dataEntry.clusterId) {
                continue;
            }

            dataEntry.clusterId = newClusterId;
            nothingChanged = false;
        }

        if (nothingChanged) {
            break;
        }
    }

    std::cout << "INFO: Total number of iterations == " << i << std::endl;

    QFile outputFile("clustering_results");

    outputFile.open(QFile::WriteOnly);
    if (!outputFile.isOpen()) {
        std::cerr << "ERROR: Failed to open output file \"" << "clustering_results" << "\"" << std::endl;
        return -1;
    }

    QTextStream resultsStream(&outputFile);

    for (auto dataEntry : data) {
        resultsStream << QString::fromStdString(dataEntry.day) << "," << dataEntry.clusterId + 1 << "\n";
//        std::cout << dataEntry.day << "," << dataEntry.clusterId << "\n";
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

std::vector<std::vector<double>> randInitialCentroids(std::vector<WaterPlantDataEntry> data, int32_t hyperparameterK) {
    std::set<int32_t> entryIds;

    for (int32_t i = 0; i < hyperparameterK; i++) {
        int32_t newNumber = rand() % data.size();

        while (entryIds.find(newNumber) != entryIds.end()) {
            newNumber = rand() % data.size();
        }

        entryIds.insert(newNumber);
    }

    std::vector<std::vector<double>> centroids(hyperparameterK);

    for (int32_t i = 0; i < hyperparameterK; i++) {

        centroids[i] = data[*(std::next(entryIds.begin(), i))].features;
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

std::vector<std::vector<double>> computeCentroids(std::vector<WaterPlantDataEntry> data, int32_t hyperparameterK) {
    std::vector<std::vector<double>> newCentroids(hyperparameterK);
    for (int32_t i = 0; i < hyperparameterK; i++) {
        newCentroids[i] = std::vector<double>(numFeatures, 0);
    }

    std::vector<uint32_t> numEntriesInCluster(hyperparameterK, 0);

    for (auto dataEntry : data) {
        numEntriesInCluster[dataEntry.clusterId]++;

        for (uint32_t featureId = 0; featureId < numFeatures; featureId++) {
            newCentroids[dataEntry.clusterId][featureId] += dataEntry.features[featureId];
        }
    }

    for (int32_t clusterId = 0; clusterId < hyperparameterK; clusterId++) {
        if (numEntriesInCluster[clusterId] == 0) {
            continue;
        }

        for (uint32_t i = 0; i < numFeatures; i++) {
            newCentroids[clusterId][i] /= numEntriesInCluster[clusterId];
        }
    }

    return newCentroids;
}

void fixMissingData(std::vector<WaterPlantDataEntry>& data) {
    std::vector<double> featureMeanValues(numFeatures, 0);

    for (auto dataEntry : data) {
        for (uint32_t i = 0; i < numFeatures; i++) {
            if (!isnan(dataEntry.features[i])) {
                featureMeanValues[i] += dataEntry.features[i];
            }
        }
    }

    if (data.empty()) return;

    for (uint32_t i = 0; i < numFeatures; i++) {
            featureMeanValues[i] /= data.size();
    }

    for (auto& dataEntry : data) {
        for (uint32_t i = 0; i < numFeatures; i++) {
            if (isnan(dataEntry.features[i])) {
                dataEntry.features[i] = featureMeanValues[i];
            }
        }
    }
}

void dataNormalization(std::vector<WaterPlantDataEntry>& data) {
    std::vector<double> featureMinValues(numFeatures, std::numeric_limits<double>::max());
    std::vector<double> featureMaxValues(numFeatures, std::numeric_limits<double>::lowest());

    for (auto dataEntry : data) {
        for (uint32_t i = 0; i < numFeatures; i++) {
            if (dataEntry.features[i] < featureMinValues[i]) {
                featureMinValues[i] = dataEntry.features[i];
            }
            if (dataEntry.features[i] > featureMaxValues[i]) {
                featureMaxValues[i] = dataEntry.features[i];
            }
        }
    }

    for (auto& dataEntry : data) {
        for (uint32_t i = 0; i < numFeatures; i++) {
            dataEntry.features[i] = (dataEntry.features[i] - featureMinValues[i]) / (featureMaxValues[i] - featureMinValues[i]);
        }
    }
}
