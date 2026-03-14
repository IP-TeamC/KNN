package de.fhdw.knn.network.neuron;

import de.fhdw.knn.network.Network;
import de.fhdw.knn.network.activation.ActivationFunction;
import de.fhdw.knn.network.connection.Connection;
import de.fhdw.knn.network.connection.WeightInitializer;
import de.fhdw.knn.network.layer.DenseLayer;
import de.fhdw.knn.network.layer.InputLayer;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import javax.annotation.processing.Generated;

import static org.junit.jupiter.api.Assertions.*;

public class NeuronTest {

    @Test
    public void testDenseCompute() {
        DenseNeuron dn = new DenseNeuron();
        dn.activationFunction = ActivationFunction.SIGMOID;
        dn.incoming = new Connection[]{
                new Connection(2),
                new Connection(1, false),
                new Connection(-3)
        };

        OutputDerived compute = dn.compute(new double[]{7, 5, 4});

        assertEquals(0.880797, compute.output(), 1e-6);
        assertEquals(0.104994, compute.derived(), 1e-6);
    }

    @Test
    public void testOutputsDerived() {
        double[][] outputs = new double[][]{new double[]{42, 123}};
        OutputsDerived outputsDerived = new OutputsDerived(outputs, null);
        assertEquals(outputs[0], outputsDerived.lastOutput());
    }

    private Network superNeuronInner;
    private static final double[] superNeuronAdapterBias = new double[]{0, 0, 0};
    private static final double[][] superNeuronAdapterWeights = new double[][]{
            new double[]{0, 1},
            new double[]{0, 0},
            new double[]{1, 0},
    };

    @BeforeEach
    public void initSuperNeuronInnerNetwork() {
        superNeuronInner = new Network(42, WeightInitializer.ZERO, 3,
                DenseLayer.createLayers(null, ActivationFunction.SIGMOID, 1));
        superNeuronInner.denseLayers[0].neurons[0].incoming[0].weight = 5;
        superNeuronInner.denseLayers[0].neurons[0].incoming[1].weight = 2;
        superNeuronInner.denseLayers[0].neurons[0].incoming[2].weight = 8;
    }

    @Test
    public void testSuperNeuron() {
        SuperNeuron sn = new SuperNeuron(superNeuronInner, superNeuronAdapterBias, superNeuronAdapterWeights);
        sn.incoming = new Connection[]{
                new Connection(1),
                new Connection(1)
        };
        OutputDerived compute = sn.compute(new double[]{-2, 3});
        assertEquals(0.268941, compute.output(), 1e-6);
        assertEquals(0.196612, compute.derived(), 1e-6);

        SuperNeuron clone = new SuperNeuron(sn.network);
        clone.incoming = new Connection[]{
                new Connection(1),
                new Connection(1)
        };
        OutputDerived computeClone = clone.compute(new double[]{-2, 3});
        assertEquals(0.268941, computeClone.output(), 1e-6);
        assertEquals(0.196612, computeClone.derived(), 1e-6);
    }

    @Test
    public void testSuperNeuronGuard() {
        SuperNeuron sn = new SuperNeuron(superNeuronInner, superNeuronAdapterBias, superNeuronAdapterWeights);
        sn.incoming = new Connection[]{
                new Connection(1, false),
                new Connection(1)
        };
        OutputDerived compute = sn.compute(new double[]{-2, 3});
        assertEquals(1, compute.output(), 1e-6);
        assertEquals(0, compute.derived(), 1e-6);
    }

    @Test
    public void testSuperNeuronInit() {
        Network twoOutputs = new Network(new InputLayer(0), new DenseLayer[]{new DenseLayer(2)});
        assertThrowsExactly(IllegalArgumentException.class, () -> new SuperNeuron(twoOutputs, superNeuronAdapterBias, superNeuronAdapterWeights));
        assertThrowsExactly(IllegalArgumentException.class, () -> new SuperNeuron(superNeuronInner, new double[]{0, 0}, superNeuronAdapterWeights));
        assertThrowsExactly(IllegalArgumentException.class, () -> new SuperNeuron(superNeuronInner, superNeuronAdapterBias, new double[2][]));
    }

    @Test
    public void testSuperNeuronInsert() {
        Network outer = new Network(42, WeightInitializer.ZERO, 2,
                new DenseLayer(1).withActivationFunction(ActivationFunction.LINEAR));
        outer.denseLayers[0].neurons[0].incoming[0].weight = 1;
        outer.denseLayers[0].neurons[0].incoming[1].weight = 1;

        SuperNeuron sn = new SuperNeuron(superNeuronInner, superNeuronAdapterBias, superNeuronAdapterWeights);
        sn.insert(outer, 0, 0);

        OutputsDerived feedForward = outer.feedForward(new double[]{-2, 3});
        assertEquals(0.268941, feedForward.lastOutput()[0], 1e-6);
        assertEquals(0.196612, feedForward.derived()[0][0], 1e-6);
    }

    @Test
    public void testSuperNeuronInsertComplex() {
        Network outer = new Network(42, WeightInitializer.ZERO, 2,
                new DenseLayer(1).withActivationFunction(ActivationFunction.LINEAR),
                new DenseLayer(1).withActivationFunction(ActivationFunction.LINEAR));
        outer.denseLayers[0].neurons[0].incoming[0].weight = 1;
        outer.denseLayers[0].neurons[0].incoming[1].weight = 1;
        outer.denseLayers[1].neurons[0].incoming[0].weight = 1;

        SuperNeuron sn = new SuperNeuron(superNeuronInner, superNeuronAdapterBias, superNeuronAdapterWeights);
        sn.insert(outer, 0, 0);

        OutputsDerived feedForward = outer.feedForward(new double[]{-2, 3});
        assertEquals(0.268941, feedForward.lastOutput()[0], 1e-6);
        assertEquals(0.196612, feedForward.derived()[0][0], 1e-6);
    }

    @Generated("GitHub Copilot")
    @Test
    public void testSuperNeuronInsertIntoOutputLayer() {
        Network outer = new Network(42, WeightInitializer.ZERO, 2,
                new DenseLayer(1).withActivationFunction(ActivationFunction.LINEAR));
        outer.denseLayers[0].neurons[0].incoming[0].weight = 1;
        outer.denseLayers[0].neurons[0].incoming[1].weight = 1;

        SuperNeuron sn = new SuperNeuron(superNeuronInner, superNeuronAdapterBias, superNeuronAdapterWeights);
        sn.insert(outer, 0, 0);

        OutputsDerived feedForward = outer.feedForward(new double[]{-2, 3});
        assertEquals(0.268941, feedForward.lastOutput()[0], 1e-6);
        assertEquals(0.196612, feedForward.derived()[0][0], 1e-6);
    }

}
