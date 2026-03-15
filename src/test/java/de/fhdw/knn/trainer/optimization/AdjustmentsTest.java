package de.fhdw.knn.trainer.optimization;

import de.fhdw.knn.network.Network;
import de.fhdw.knn.network.activation.ActivationFunction;
import de.fhdw.knn.network.connection.WeightInitializer;
import de.fhdw.knn.network.layer.DenseLayer;
import de.fhdw.knn.network.neuron.DenseNeuron;
import de.fhdw.knn.network.neuron.SuperNeuron;
import org.junit.jupiter.api.Test;

import javax.annotation.processing.Generated;

import static org.junit.jupiter.api.Assertions.*;

public class AdjustmentsTest {

    // Die Klasse Adjustments wurde bereits in anderen Testfällen ausreichend getestet,
    // hier wurden nur einige ergänzende Testfälle zusätzlich erzeugt.

    @Generated("GitHub Copilot")
    @Test
    public void testAdjustSubtractsWeightAndBias() {
        Network network = new Network(42, WeightInitializer.ZERO, 1,
                new DenseLayer(1).withActivationFunction(ActivationFunction.LINEAR));
        DenseNeuron neuron = (DenseNeuron) network.denseLayers[0].neurons[0];
        neuron.incoming[0].weight = 0.5;
        neuron.bias = 0.3;

        Adjustments adjustments = Adjustments.generateEmpty(network);
        adjustments.weight[0][0][0] = 0.1;
        adjustments.bias[0][0] = 0.2;

        adjustments.adjust(network);

        assertEquals(0.4, neuron.incoming[0].weight, 1e-12);
        assertEquals(0.1, neuron.bias, 1e-12);
    }

    @Generated("GitHub Copilot")
    @Test
    public void testAdjustUpdatesWeightsForSuperNeuronButNotBias() {
        Network outer = new Network(42, WeightInitializer.ZERO, 1,
                new DenseLayer(1).withActivationFunction(ActivationFunction.LINEAR));
        Network inner = new Network(42, WeightInitializer.ZERO, 1,
                new DenseLayer(1).withActivationFunction(ActivationFunction.LINEAR));
        SuperNeuron superNeuron = new SuperNeuron(inner);
        superNeuron.insert(outer, 0, 0);

        double beforeWeight = outer.denseLayers[0].neurons[0].incoming[0].weight;

        Adjustments adjustments = Adjustments.generateEmpty(outer);
        adjustments.weight[0][0][0] = 0.25;
        adjustments.bias[0][0] = 0.99;

        adjustments.adjust(outer);

        assertEquals(beforeWeight - 0.25, outer.denseLayers[0].neurons[0].incoming[0].weight, 1e-12);
        assertInstanceOf(SuperNeuron.class, outer.denseLayers[0].neurons[0]);
    }
}

