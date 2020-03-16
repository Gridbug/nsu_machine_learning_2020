#include <iostream>
#include <vector>
#include <algorithm>

#include <QDataStream>
#include <QFile>
#include <QTextStream>

int main(int argc, char* argv[]) {
    if (2 != argc) {
        std::cerr << "ERROR: Wrong number of arguments. Usage " << argv[0] << " <data_file_name>" << std::endl;
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
    std::vector<std::pair<std::string, std::vector<double>>> data;

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

    if (data.empty()) {
        std::cerr << "ERROR: Not found any data in file \"" << argv[1] << "\"" << std::endl;
        return -1;
    }

    std::cout << "INFO: Found " << data.size() << " data lines and successfully parsed them." << std::endl;

    dataFile.close();

    return 0;
}
