package de.fhdw.knn.data;

import org.junit.jupiter.api.Test;

import javax.annotation.processing.Generated;
import java.util.Arrays;

import static org.junit.jupiter.api.Assertions.*;

public class DataSetTest {

    private DataSet setUpDataSet() {
        double[][] input = new double[12][5];
        double[][] output = new double[12][1];
        for (int i = 0; i < input.length; i++) {
            for (int j = 0; j < 5; j++) {
                input[i][j] = Math.random();
                output[i][0] = Math.random();
            }
        }
        return new DataSet(input, output);
    }

    @Test
    public void testUnevenDataSetConstruction() {
        assertThrows(IllegalArgumentException.class, () ->
                new DataSet(new double[12][3], new double[16][2]));
    }

    @Test
    public void testPreprocessDoesSomething() {
        double[][] input = new double[12][1];
        double[][] output = new double[12][1];
        for (int i = 0; i < input.length; i++) {
            input[i][0] = Math.random();
            output[i][0] = Math.random();
        }
        DataSet dataSet = new DataSet(input, output);
        double[][] inputsBefore = new double[input.length][];
        for (int i = 0; i < input.length; i++) {
            inputsBefore[i] = dataSet.inputs[i].clone();
        }

        dataSet.preprocess((i, o) -> i[0] += o[0]);

        assertFalse(Arrays.deepEquals(inputsBefore, dataSet.inputs));
    }

    @Test
    public void testSubsetInputs() {
        DataSet dataSet = this.setUpDataSet();
        DataSet dataSubSet = dataSet.subsetInputs(3, 4);

        for (int i = 0; i < dataSet.size; i++) {
            assertArrayEquals(new double[]{dataSet.inputs[i][3], dataSet.inputs[i][4]}, dataSubSet.inputs[i]);
        }
    }

    @Test
    public void testShuffle() {
        DataSet dataSet = this.setUpDataSet();
        double[][] inputsBefore = new double[dataSet.inputs.length][];
        for (int i = 0; i < dataSet.inputs.length; i++) {
            inputsBefore[i] = dataSet.inputs[i].clone();
        }
        dataSet.shuffle(42);
        assertFalse(Arrays.deepEquals(inputsBefore, dataSet.inputs));
    }

    @Generated("GitHub Copilot")
    @Test
    public void testShuffleAndSplitWithNormalTestShare() {
        DataSet dataSet = this.setUpDataSet();
        double[][] inputsBefore = new double[dataSet.inputs.length][];
        for (int i = 0; i < dataSet.inputs.length; i++) {
            inputsBefore[i] = dataSet.inputs[i].clone();
        }
        TrainTestSplit trainTestSplit = dataSet.shuffleAndSplit(42, 0.3);
        assertEquals(trainTestSplit.test.size, (int) (dataSet.size * 0.3));
        assertEquals(trainTestSplit.train.size, (dataSet.size - ((int) (dataSet.size * 0.3))));
        for (double[] input : dataSet.inputs) {
            assertTrue(Arrays.stream(inputsBefore).anyMatch(x -> Arrays.equals(x, input)));
        }
    }

    @Test
    public void testShuffleAndSplitWithZeroAndOneTestShare() {
        DataSet dataSet = this.setUpDataSet();
        TrainTestSplit zeroTestShare = dataSet.shuffleAndSplit(42, 0);
        TrainTestSplit oneTestShare = dataSet.shuffleAndSplit(42, 1);
        assertNull(zeroTestShare.test);
        assertNull(oneTestShare.train);
    }

    @Test
    void testShuffleAndSplitWithToBigTestShare() {
        DataSet dataSet = this.setUpDataSet();
        IllegalArgumentException exception = assertThrows(IllegalArgumentException.class, () ->
                dataSet.shuffleAndSplit(42, 1.2));
        assertEquals("Test share must be between 0 and 1", exception.getMessage());
    }

    @Test
    void testShuffleAndSplitWithToLittleTestShare() {
        DataSet dataSet = this.setUpDataSet();
        IllegalArgumentException exception = assertThrows(IllegalArgumentException.class, () ->
                dataSet.shuffleAndSplit(42, -0.2));
        assertEquals("Test share must be between 0 and 1", exception.getMessage());
    }

    @Test
    public void testNormalizeAndDenormalizeInputs() {
        DataSet dataSet = this.setUpDataSet();
        double min = 1.0;
        double max = 3.0;
        Normalizer normalizer = new MinMaxNormalizer(min, max);
        dataSet.normalizeInputs(normalizer);
        for (double[] row : dataSet.inputs) {
            assertTrue(Arrays.stream(row).allMatch(x -> x >= min && x <= max)); //Previously randomly initialized from 0 to 1
        }
        dataSet.denormalizeInputs();
        for (double[] row : dataSet.inputs) {
            assertTrue(Arrays.stream(row).allMatch(x -> x <= 1 && x >= 0)); //Should now be back in range of 0 to 1
        }
    }

    @Test
    public void testNormalizeAndDenormalizeOutputs() {
        DataSet dataSet = this.setUpDataSet();
        double min = 1.0;
        double max = 3.0;
        Normalizer normalizer = new MinMaxNormalizer(min, max);
        dataSet.normalizeOutputs(normalizer);
        for (double[] row : dataSet.outputs) {
            assertTrue(Arrays.stream(row).allMatch(x -> x >= min && x <= max)); //Previously randomly initialized from 0 to 1
        }
        dataSet.denormalizeOutputs();
        for (double[] row : dataSet.outputs) {
            assertTrue(Arrays.stream(row).allMatch(x -> x <= 1 && x >= 0)); //Should now be back in range of 0 to 1
        }
    }
}
