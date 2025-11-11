package de.fhdw.knn.data;

import java.io.*;
import java.util.Arrays;
import java.util.List;
import java.util.Map;

public class CsvReader {

    public static DataSet readFile(String fileName, int inputStart, int inputSize, int outputStart, int outputSize)  throws IOException{
        return readFile(fileName, inputStart, inputSize, outputStart, outputSize, 0);
    }

    public static DataSet readFile(String fileName, int inputStart, int inputSize, int outputStart, int outputSize, final int skip) throws IOException {
        try (BufferedReader br = new BufferedReader(new FileReader(fileName))) {
            List<Map.Entry<double[], double[]>> data = br.lines().skip(skip).map(line -> line.split(",")).map(split -> {
                double[] dataset = Arrays.stream(split).mapToDouble(CsvReader::parseDoubleOrNaN).toArray();
                double[] input = new double[inputSize];
                double[] output = new double[outputSize];
                System.arraycopy(dataset, inputStart, input, 0, inputSize);
                System.arraycopy(dataset, outputStart, output, 0, outputSize);
                return Map.entry(input, output);
            }).toList();

            double[][] input = new double[data.size()][];
            double[][] output = new double[data.size()][];
            for (int i = 0; i < data.size(); i++) {
                input[i] = data.get(i).getKey();
                output[i] = data.get(i).getValue();
            }
            return new DataSet(input, output);
        }
    }

    private static double parseDoubleOrNaN(String value) {
        try {
            return Double.parseDouble(value);
        } catch (NumberFormatException e) {
            return Double.NaN;
        }
    }

}
