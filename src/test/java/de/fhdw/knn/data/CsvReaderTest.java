package de.fhdw.knn.data;

import org.junit.jupiter.api.Test;

import javax.annotation.processing.Generated;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.function.Function;
import java.util.stream.Stream;

import static org.junit.jupiter.api.Assertions.*;

public class CsvReaderTest {

    @Generated("GitHub Copilot")
    Function<String, Stream<Map.Entry<double[], double[]>>> lineParser = line -> {
        String[] parts = line.split(",");
        double[] input = {Double.parseDouble(parts[0])};
        double[] output = {Double.parseDouble(parts[1])};
        return Stream.of(Map.entry(input, output));
    };

    @Test
    public void testCsvReader() throws IOException {
        DataSet dataSet = CsvReader.readFile("data/scorer_training_dataset.csv", 0, 1, 1, 1, 1);

        assertEquals(502, dataSet.size);
        assertEquals(1, dataSet.inputSize);
        assertEquals(1, dataSet.outputSize);
        assertArrayEquals(new String[]{"actual"}, dataSet.inputLabels);
        assertArrayEquals(new String[]{"predicted"}, dataSet.outputLabels);
        double[][] actualInputs = new double[502][1];
        double[][] actualOutputs = new double[502][1];
        assertEquals(actualInputs.length, dataSet.inputs.length);
        assertEquals(actualOutputs.length, dataSet.outputs.length);
    }

    @Test
    public void testEmptyCsv() {
        IllegalArgumentException exception = assertThrows(IllegalArgumentException.class, () -> {
            DataSet dataSet = CsvReader.readFile("data/empty.csv", 0, 1, 1, 1);
        });
        assertEquals("Csv-File not allowed to be empty.", exception.getMessage());
    }

    @Test
    void testComplexReadFileWithoutLabelParser() throws IOException {
        DataSet result = CsvReader.readFile("data/scorer_training_dataset.csv", 1, lineParser);

        assertEquals(502, result.size);
        assertArrayEquals(null, result.inputLabels); //labelParser = null
        assertArrayEquals(null, result.outputLabels); //labelParser = null
    }

    @Generated("GitHub Copilot")
    @Test
    void testWithSkipAndLabelParser() throws IOException {
        List<String> capturedLabels = new ArrayList<>();

        DataSet result = CsvReader.readFile("data/scorer_training_dataset.csv", 1, lineParser,
                (ds, s) -> capturedLabels.add(s));

        assertNotNull(result);
        assertEquals(1, capturedLabels.size());
        assertEquals("actual,predicted", capturedLabels.get(0));
    }

    @Test
    public void testCsvReaderConstructor() {
        CsvReader csvReader = new CsvReader();
        assertNotNull(csvReader);
    }

    @Test
    void testParseDoubleOrNaNWithInvalidData() throws IOException {
        DataSet dataSet = CsvReader.readFile("data/test_invalid.csv", 0, 1, 1, 1, 1);

        assertEquals(4, dataSet.size);

        boolean hasNaN = false;
        for (double[] input : dataSet.inputs) {
            if (Arrays.stream(input).allMatch(x -> Double.isNaN(x))){
                hasNaN = true;
                break;
            }
        }
        for (double[] output : dataSet.outputs) {
            if (Arrays.stream(output).allMatch(x -> Double.isNaN(x))) {
                hasNaN = true;
                break;
            }
        }
        assertTrue(hasNaN);
    }
}
