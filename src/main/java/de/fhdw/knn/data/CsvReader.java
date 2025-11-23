package de.fhdw.knn.data;

import java.io.*;
import java.util.Arrays;
import java.util.List;
import java.util.Map;

public class CsvReader {

    public static DataSet readFile(String fileName, int inputStart, int inputSize, int outputStart, int outputSize)  throws IOException{
        return readFile(fileName, inputStart, inputSize, outputStart, outputSize, 0);
    }

    public static DataSet readFile(String fileName, int inputStart, int inputSize, int outputStart, int outputSize, int skip) throws IOException {
        try (BufferedReader br = new BufferedReader(new FileReader(fileName))) {
            String[] inputLabels = null;
            String[] outputLabels = null;
            if (skip == 1) {
                inputLabels = new String[inputSize];
                outputLabels = new String[outputSize];
                String[] headers = br.readLine().split(",");
                System.arraycopy(headers, inputStart, inputLabels, 0, inputSize);
                System.arraycopy(headers, outputStart, outputLabels, 0, outputSize);
                skip = 0;
            }
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
            DataSet dataSet = new DataSet(input, output);
            if (inputLabels != null) {
                dataSet.inputLabels = inputLabels;
                dataSet.outputLabels = outputLabels;
            }
            return dataSet;
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
