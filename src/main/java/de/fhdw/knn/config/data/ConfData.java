package de.fhdw.knn.config.data;

import de.fhdw.knn.data.CsvReader;
import de.fhdw.knn.data.TrainTestSplit;
import io.github.wasabithumb.jtoml.serial.TomlSerializable;
import lombok.Data;

import java.io.IOException;

@Data
public class ConfData implements TomlSerializable {

    private String file;
    private int inputStart;
    private int inputSize;
    private int outputStart;
    private int outputSize;
    private int skip;

    private int seed;
    private double testShare;

    public TrainTestSplit create() throws IOException {
        return CsvReader.readFile(file, inputStart, inputSize, outputStart, outputSize, skip).shuffleAndSplit(seed, testShare);
    }

}
