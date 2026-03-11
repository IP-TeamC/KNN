package de.fhdw.knn.network.activation;

/**
 * Swish-Aktivierungsfunktion
 *
 * @see ActivationFunction#SWISH
 */
class SwishActivationFunction implements ActivationFunction {

    /**
     * Die ActivationFunction ist zustandslos und es sollten keine weiteren Instanzen außerhalb von {@link ActivationFunction} erzeugt werden
     */
    SwishActivationFunction() {
    }

    @Override
    public double calc(double input) {
        return input / (1 + Math.exp(-input));
    }

    @Override
    public double derived(double input, double calc) {
        return calc + (1 - calc) / (1 + Math.exp(-input));
    }
}
